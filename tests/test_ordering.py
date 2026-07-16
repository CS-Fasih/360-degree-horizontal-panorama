from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from panorama360.models import ImageQuality, PairwiseMatch
from panorama360.ordering import SequenceAnalyzer


def _strong_match(left: int, right: int, score_multiplier: float = 1.0) -> PairwiseMatch:
    return PairwiseMatch(
        left=left,
        right=right,
        good_matches=100,
        inliers=round(80 * score_multiplier),
        inlier_ratio=0.8,
        source_coverage=0.18,
        target_coverage=0.17,
        median_error=0.7,
        horizontal_shift=-100.0,
        homography=np.eye(3, dtype=np.float64),
    )


def _weak_match(left: int, right: int) -> PairwiseMatch:
    return PairwiseMatch(left=left, right=right, good_matches=4)


def test_cycle_search_recovers_circular_neighbors() -> None:
    count = 8
    matches: dict[tuple[int, int], PairwiseMatch] = {}
    ring = [0, 3, 6, 2, 7, 1, 5, 4]
    ring_edges = {
        tuple(sorted((ring[index], ring[(index + 1) % count]))) for index in range(count)
    }
    for left in range(count):
        for right in range(left + 1, count):
            matches[(left, right)] = (
                _strong_match(left, right) if (left, right) in ring_edges else _weak_match(left, right)
            )

    order = SequenceAnalyzer(beam_width=250)._find_best_cycle(count, matches)
    found_edges = {
        tuple(sorted((order[index], order[(index + 1) % count]))) for index in range(count)
    }
    assert found_edges == ring_edges


def test_validation_rejects_sequence_without_last_to_first_overlap() -> None:
    count = 6
    order = list(range(count))
    matches: dict[tuple[int, int], PairwiseMatch] = {}
    for left in range(count):
        for right in range(left + 1, count):
            is_neighbor = right == left + 1
            matches[(left, right)] = (
                _strong_match(left, right) if is_neighbor else _weak_match(left, right)
            )
    qualities = [ImageQuality(1600, 900, 200.0, 1200, 125.0) for _ in range(count)]

    _warnings, errors, closure, confidence = SequenceAnalyzer._validate(
        qualities, matches, order, auto_order=False
    )

    assert not closure
    assert confidence < 100
    assert any("insufficient overlap" in error for error in errors)


def test_soft_frames_warn_but_do_not_block_valid_overlap() -> None:
    count = 5
    order = list(range(count))
    matches: dict[tuple[int, int], PairwiseMatch] = {}
    for left in range(count):
        for right in range(left + 1, count):
            is_neighbor = right == left + 1 or (left == 0 and right == count - 1)
            matches[(left, right)] = (
                _strong_match(left, right) if is_neighbor else _weak_match(left, right)
            )
    qualities = [ImageQuality(1600, 900, 4.0, 300, 125.0) for _ in range(count)]

    warnings, errors, closure, _confidence = SequenceAnalyzer._validate(
        qualities, matches, order, auto_order=False
    )

    assert closure
    assert not errors
    assert any("look soft" in warning for warning in warnings)


def _periodic_texture(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(2026)
    image = np.full((height, width, 3), 218, dtype=np.uint8)
    for _ in range(650):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(2, 11))
        color = tuple(int(value) for value in rng.integers(10, 245, size=3))
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    for index in range(28):
        x = index * width // 28
        cv2.putText(
            image,
            str(index),
            (x + 3, 35 + (index % 4) * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
    return image


def test_real_feature_analysis_reorders_periodic_photo_set(tmp_path: Path) -> None:
    world_width, height = 1600, 420
    frame_width = 500
    count = 8
    texture = _periodic_texture(world_width, height)
    tiled = np.concatenate([texture, texture], axis=1)
    paths: list[Path] = []
    for index in range(count):
        start = index * (world_width // count)
        frame = tiled[:, start : start + frame_width]
        path = tmp_path / f"frame_{index}.jpg"
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, quality=95)
        paths.append(path)

    scrambled = [paths[index] for index in [0, 4, 2, 7, 1, 5, 3, 6]]
    analysis = SequenceAnalyzer(max_dimension=900, beam_width=300).analyze(scrambled)
    ordered_original_indices = [int(path.stem.split("_")[-1]) for path in analysis.ordered_paths]
    circular_steps = {
        min((b - a) % count, (a - b) % count)
        for a, b in zip(ordered_original_indices, ordered_original_indices[1:] + ordered_original_indices[:1])
    }

    assert circular_steps == {1}
    assert analysis.has_circular_closure
    assert not analysis.errors
