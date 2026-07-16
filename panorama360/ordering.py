"""Photo quality analysis, all-pairs overlap detection, and circular ordering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .image_io import load_rgb
from .models import (
    AnalysisResult,
    ImageQuality,
    PairwiseMatch,
    PanoramaError,
    ProgressCallback,
)


@dataclass(slots=True)
class _Features:
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    scale: float


class SequenceAnalyzer:
    """Find a dependable circular image sequence without assuming image count."""

    def __init__(self, max_dimension: int = 1400, beam_width: int = 600) -> None:
        self.max_dimension = max_dimension
        self.beam_width = beam_width

    def analyze(
        self,
        paths: Iterable[str | Path],
        *,
        auto_order: bool = True,
        progress: ProgressCallback | None = None,
    ) -> AnalysisResult:
        sources = [Path(path).expanduser().resolve() for path in paths]
        if len(sources) < 3:
            raise PanoramaError(
                "More photos are needed",
                "A 360-degree panorama needs at least three overlapping photos.",
                ("Select every original photo from one complete rotation.",),
            )
        if len(sources) > 80:
            raise PanoramaError(
                "Too many photos",
                f"{len(sources)} photos were selected. This application supports up to 80 at once.",
                ("Remove burst duplicates and keep one continuous rotation.",),
            )

        callback = progress or (lambda _value, _message: None)
        callback(1, "Loading photos…")
        feature_sets: list[_Features] = []
        qualities: list[ImageQuality] = []
        sift = cv2.SIFT_create(nfeatures=3200, contrastThreshold=0.025, edgeThreshold=12)

        for index, source in enumerate(sources):
            rgb = load_rgb(source)
            scaled, scale = self._analysis_image(rgb)
            gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
            # CLAHE only guides feature detection; original pixels remain untouched.
            feature_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            keypoints, descriptors = sift.detectAndCompute(feature_gray, None)
            keypoints = keypoints or []
            qualities.append(
                ImageQuality(
                    width=int(rgb.shape[1]),
                    height=int(rgb.shape[0]),
                    blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    feature_count=len(keypoints),
                    mean_luminance=float(gray.mean()),
                )
            )
            feature_sets.append(_Features(scaled, keypoints, descriptors, scale))
            callback(
                3 + round(25 * (index + 1) / len(sources)),
                f"Finding details in photo {index + 1} of {len(sources)}…",
            )

        matches: dict[tuple[int, int], PairwiseMatch] = {}
        pair_count = len(sources) * (len(sources) - 1) // 2
        completed = 0
        for left in range(len(sources)):
            for right in range(left + 1, len(sources)):
                matches[(left, right)] = self._match_pair(
                    left, right, feature_sets[left], feature_sets[right]
                )
                completed += 1
                callback(
                    28 + round(57 * completed / pair_count),
                    f"Comparing overlaps {completed} of {pair_count}…",
                )

        callback(88, "Finding the circular photo order…")
        order = self._find_best_cycle(len(sources), matches) if auto_order else list(range(len(sources)))
        order = self._normalize_direction(order, matches)
        warnings, errors, closure, confidence = self._validate(
            qualities, matches, order, auto_order=auto_order
        )
        callback(100, "Analysis complete")
        return AnalysisResult(
            paths=sources,
            order=order,
            matches=matches,
            qualities=qualities,
            warnings=warnings,
            errors=errors,
            confidence=confidence,
            has_circular_closure=closure,
        )

    def _analysis_image(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = rgb.shape[:2]
        scale = min(1.0, self.max_dimension / max(height, width))
        if scale >= 0.999:
            return rgb, 1.0
        resized = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _match_pair(left: int, right: int, a: _Features, b: _Features) -> PairwiseMatch:
        result = PairwiseMatch(left=left, right=right)
        if a.descriptors is None or b.descriptors is None:
            return result
        if len(a.descriptors) < 12 or len(b.descriptors) < 12:
            return result

        # A KD-tree matcher keeps all-pairs analysis responsive for 18–36
        # photos while preserving SIFT's accuracy on exposure changes.
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=64),
        )
        try:
            forward_knn = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
            reverse_knn = matcher.knnMatch(b.descriptors, a.descriptors, k=2)
        except cv2.error:
            return result

        forward = [m for pair in forward_knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.74 * n.distance]
        reverse_pairs = {
            (m.trainIdx, m.queryIdx)
            for pair in reverse_knn
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.74 * n.distance
        }
        mutual = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_pairs]
        good = mutual if len(mutual) >= 12 else forward
        result.good_matches = len(good)
        if len(good) < 8:
            return result

        source_points = np.float32([a.keypoints[m.queryIdx].pt for m in good])
        target_points = np.float32([b.keypoints[m.trainIdx].pt for m in good])
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        try:
            homography, mask = cv2.findHomography(
                source_points,
                target_points,
                method,
                ransacReprojThreshold=4.0,
                maxIters=10000,
                confidence=0.999,
            )
        except cv2.error:
            return result
        if homography is None or mask is None or not np.isfinite(homography).all():
            return result

        inlier_mask = mask.ravel().astype(bool)
        inlier_source = source_points[inlier_mask]
        inlier_target = target_points[inlier_mask]
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 4:
            return result

        projected = cv2.perspectiveTransform(inlier_source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - inlier_target, axis=1)
        shifts = inlier_target - inlier_source
        result.inliers = inlier_count
        result.inlier_ratio = inlier_count / max(1, len(good))
        result.source_coverage = SequenceAnalyzer._point_coverage(inlier_source, a.image.shape)
        result.target_coverage = SequenceAnalyzer._point_coverage(inlier_target, b.image.shape)
        result.median_error = float(np.median(errors))
        result.horizontal_shift = float(np.median(shifts[:, 0]))
        result.vertical_shift = float(np.median(shifts[:, 1]))
        result.homography = homography
        return result

    @staticmethod
    def _point_coverage(points: np.ndarray, image_shape: tuple[int, ...]) -> float:
        if len(points) < 3:
            return 0.0
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
        return area / max(1.0, float(image_shape[0] * image_shape[1]))

    @staticmethod
    def _edge_utility(a: int, b: int, matches: dict[tuple[int, int], PairwiseMatch]) -> float:
        match = matches[(min(a, b), max(a, b))]
        if match.is_usable:
            return math.log1p(max(0.0, match.score))
        # A cycle with a missing edge must always lose to one made entirely of
        # real overlaps, even if it contains several exceptionally strong pairs.
        return -12.0 + min(1.0, match.inliers / 20.0)

    def _find_best_cycle(
        self, count: int, matches: dict[tuple[int, int], PairwiseMatch]
    ) -> list[int]:
        if count == 3:
            return [0, 1, 2]

        # A fixed start removes equivalent rotations. Beam search retains
        # several plausible sequences where greedy nearest-neighbour ordering
        # would commit too early in scenes containing repeated windows/trees.
        beam: list[tuple[float, tuple[int, ...], int]] = [(0.0, (0,), 1)]
        for _depth in range(1, count):
            candidates: list[tuple[float, tuple[int, ...], int]] = []
            for score, path, used in beam:
                last = path[-1]
                for node in range(1, count):
                    bit = 1 << node
                    if used & bit:
                        continue
                    utility = self._edge_utility(last, node, matches)
                    candidates.append((score + utility, path + (node,), used | bit))
            candidates.sort(key=lambda item: item[0], reverse=True)
            beam = candidates[: self.beam_width]

        best = max(
            beam,
            key=lambda item: item[0] + self._edge_utility(item[1][-1], item[1][0], matches),
        )
        return self._two_opt(list(best[1]), matches)

    def _two_opt(
        self, order: list[int], matches: dict[tuple[int, int], PairwiseMatch]
    ) -> list[int]:
        count = len(order)
        improved = True
        passes = 0
        while improved and passes < 8:
            improved = False
            passes += 1
            for first in range(1, count - 1):
                for last in range(first + 1, count):
                    a, b = order[first - 1], order[first]
                    c, d = order[last], order[(last + 1) % count]
                    old = self._edge_utility(a, b, matches) + self._edge_utility(c, d, matches)
                    new = self._edge_utility(a, c, matches) + self._edge_utility(b, d, matches)
                    if new > old + 1e-8:
                        order[first : last + 1] = reversed(order[first : last + 1])
                        improved = True
        return order

    @staticmethod
    def _normalize_direction(
        order: list[int], matches: dict[tuple[int, int], PairwiseMatch]
    ) -> list[int]:
        if not order:
            return order
        # Keep the first selected photo at the beginning for a stable preview.
        if 0 in order:
            pivot = order.index(0)
            order = order[pivot:] + order[:pivot]

        shifts: list[float] = []
        for a, b in zip(order[:-1], order[1:]):
            match = matches[(min(a, b), max(a, b))]
            shift = match.horizontal_shift if a < b else -match.horizontal_shift
            if match.is_usable:
                shifts.append(shift)
        # In a conventional left-to-right camera sweep, shared content moves
        # left in the next frame. Reverse only the tail so item zero stays first.
        if shifts and float(np.median(shifts)) > 0:
            order = [order[0], *reversed(order[1:])]
        return order

    @staticmethod
    def _validate(
        qualities: list[ImageQuality],
        matches: dict[tuple[int, int], PairwiseMatch],
        order: list[int],
        *,
        auto_order: bool,
    ) -> tuple[list[str], list[str], bool, float]:
        warnings: list[str] = []
        errors: list[str] = []

        weak_features = [index + 1 for index, item in enumerate(qualities) if item.feature_count < 45]
        if weak_features:
            errors.append(
                "Not enough recognizable detail was found in photo(s) "
                + ", ".join(map(str, weak_features))
                + "."
            )
        soft_photos = [index + 1 for index, item in enumerate(qualities) if item.blur_score < 55.0]
        if soft_photos:
            warnings.append(
                "Photo(s) "
                + ", ".join(map(str, soft_photos))
                + " look soft; tolerant feature matching and seam blending will be used."
            )

        aspects = np.array([item.width / item.height for item in qualities], dtype=np.float64)
        if float(aspects.max() / aspects.min()) > 1.25:
            errors.append("The photos have incompatible orientations or aspect ratios.")
        elif float(aspects.max() / aspects.min()) > 1.08:
            warnings.append("The photo dimensions vary; some output resolution may be lost during cropping.")

        luminance = np.array([item.mean_luminance for item in qualities], dtype=np.float64)
        if float(luminance.max() - luminance.min()) > 65:
            warnings.append("Large exposure differences were detected; brightness compensation will be applied.")

        edges: list[PairwiseMatch] = []
        for position, a in enumerate(order):
            b = order[(position + 1) % len(order)]
            edges.append(matches[(min(a, b), max(a, b))])
        usable = [edge for edge in edges if edge.is_usable]

        if len(usable) != len(edges):
            missing_positions = [
                f"{position + 1}→{(position + 1) % len(order) + 1}"
                for position, edge in enumerate(edges)
                if not edge.is_usable
            ]
            prefix = "The detected sequence" if auto_order else "The current manual sequence"
            errors.append(
                f"{prefix} has insufficient overlap at " + ", ".join(missing_positions) + "."
            )

        closure_edge = edges[-1]
        edge_scores = [edge.score for edge in edges if edge.is_usable]
        median_score = float(np.median(edge_scores)) if edge_scores else 0.0
        closure = (
            len(usable) == len(edges)
            and closure_edge.score >= max(0.8, median_score * 0.25)
        )
        if len(usable) == len(edges) and not closure:
            errors.append(
                "The first and last photos do not overlap strongly enough to prove complete 360-degree coverage."
            )

        if usable:
            large_vertical = [edge for edge in usable if abs(edge.vertical_shift) > 0.28 * qualities[0].height]
            if large_vertical:
                warnings.append("The camera height changed noticeably; automatic horizon correction will be applied.")
            noisy = [edge for edge in usable if edge.median_error > 4.0]
            if noisy:
                warnings.append("Possible parallax or moving objects were detected in some overlaps.")

        confidence = 0.0
        if edges:
            valid_fraction = len(usable) / len(edges)
            median_strength = min(1.0, median_score / 14.0)
            weakest = min((edge.score for edge in edges), default=0.0)
            consistency = min(1.0, weakest / max(1e-6, median_score * 0.45)) if median_score else 0.0
            confidence = round(100.0 * valid_fraction * median_strength * consistency, 1)
        return warnings, errors, closure, confidence
