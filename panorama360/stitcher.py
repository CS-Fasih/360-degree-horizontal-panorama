"""High-quality cylindrical panorama composition using OpenCV's detail API."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .image_io import load_rgb
from .models import AnalysisResult, PanoramaError, ProgressCallback, StitchResult
from .ordering import SequenceAnalyzer


class CylindricalStitcher:
    """Register, calibrate, warp, seam, and blend a circular photo sequence."""

    def __init__(
        self,
        *,
        work_megapixels: float = 0.8,
        seam_megapixels: float = 0.16,
        compose_megapixels: float = 12.0,
        blend_strength: float = 7.0,
    ) -> None:
        self.work_megapixels = work_megapixels
        self.seam_megapixels = seam_megapixels
        self.compose_megapixels = compose_megapixels
        self.blend_strength = blend_strength
        self.analyzer = SequenceAnalyzer()

    def create(
        self,
        paths: Iterable[str | Path],
        *,
        auto_order: bool = True,
        progress: ProgressCallback | None = None,
    ) -> StitchResult:
        """Analyze and stitch paths into one validated 360° cylindrical image."""

        callback = progress or (lambda _value, _message: None)

        def analysis_progress(value: int, message: str) -> None:
            callback(round(value * 0.34), message)

        analysis = self.analyzer.analyze(
            paths,
            auto_order=auto_order,
            progress=analysis_progress,
        )
        if not analysis.can_stitch:
            detail = "\n".join(f"• {item}" for item in analysis.errors)
            raise PanoramaError(
                "The photos cannot form a reliable 360° panorama",
                detail or "The overlap analysis did not confirm a complete circular sequence.",
                (
                    "Use all photos from one continuous rotation, with 25–50% overlap.",
                    "Remove blurry photos and photos taken from a different position.",
                    "Reorder the thumbnails manually if the automatic order is incorrect.",
                ),
            )

        ordered_paths = analysis.ordered_paths

        def stitch_progress(value: int, message: str) -> None:
            callback(34 + round(value * 0.66), message)

        image_rgb, field_of_view, final_warnings = self._compose(
            ordered_paths,
            progress=stitch_progress,
        )
        callback(100, "Panorama complete")
        return StitchResult(
            image_rgb=image_rgb,
            analysis=analysis,
            output_width=int(image_rgb.shape[1]),
            output_height=int(image_rgb.shape[0]),
            field_of_view_degrees=field_of_view,
            warnings=[*analysis.warnings, *final_warnings],
        )

    def _compose(
        self,
        paths: list[Path],
        *,
        progress: ProgressCallback,
    ) -> tuple[np.ndarray, float, list[str]]:
        progress(1, "Loading full-resolution photos…")
        full_images = [cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2BGR) for path in paths]
        full_sizes = [(image.shape[1], image.shape[0]) for image in full_images]
        reference_area = max(height * width for height, width in (image.shape[:2] for image in full_images))
        work_scale = self._megapixel_scale(reference_area, self.work_megapixels)
        seam_scale = self._megapixel_scale(reference_area, self.seam_megapixels)
        seam_work_aspect = seam_scale / work_scale

        # More features are not always better here: dense repeated texture can
        # produce low-confidence second-neighbour edges that destabilize loop
        # bundle adjustment. Spatially distributed top responses are enough.
        finder = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.025, edgeThreshold=12)
        features = []
        seam_images: list[np.ndarray] = []
        for index, full_image in enumerate(full_images):
            work_image = self._resize(full_image, work_scale)
            features.append(cv2.detail.computeImageFeatures2(finder, work_image))
            seam_images.append(self._resize(full_image, seam_scale))
            progress(
                3 + round(17 * (index + 1) / len(paths)),
                f"Calibrating photo {index + 1} of {len(paths)}…",
            )

        progress(22, "Matching neighboring photos and closing the 360° loop…")
        matcher = cv2.detail_BestOf2NearestMatcher(False, 0.55)
        matching_mask = self._circular_matching_mask(len(paths))
        try:
            pairwise_matches = matcher.apply2(features, matching_mask)
        finally:
            matcher.collectGarbage()

        estimator = cv2.detail_HomographyBasedEstimator()
        ok, cameras = estimator.apply(features, pairwise_matches, None)
        if not ok or len(cameras) != len(paths):
            raise PanoramaError(
                "Camera alignment failed",
                "The camera motion could not be estimated consistently from the overlaps.",
                ("Use photos taken by rotating from one fixed position.", "Increase overlap between neighboring photos."),
            )
        for camera in cameras:
            camera.R = camera.R.astype(np.float32)

        progress(31, "Optimizing alignment across the complete rotation…")
        adjuster = cv2.detail_BundleAdjusterRay()
        # OpenCV confidence is not an inlier ratio; good SIFT panorama edges
        # normally score above 1.0. Keeping the official 1.0 cutoff prevents
        # weak second-neighbour matches from corrupting the circular solution.
        adjuster.setConfThresh(1.0)
        refinement = np.zeros((3, 3), np.uint8)
        refinement[0, 0] = 1  # focal length
        refinement[0, 2] = 1  # principal point x
        refinement[1, 1] = 1  # aspect ratio
        refinement[1, 2] = 1  # principal point y
        adjuster.setRefinementMask(refinement)
        ok, cameras = adjuster.apply(features, pairwise_matches, cameras)
        if not ok:
            raise PanoramaError(
                "Alignment optimization failed",
                "The overlaps disagree too much to produce a clean panorama.",
                (
                    "Avoid moving sideways between photos.",
                    "Remove frames containing severe motion blur or mostly moving subjects.",
                ),
            )

        rotations = cv2.detail.waveCorrect(
            [np.copy(camera.R) for camera in cameras],
            cv2.detail.WAVE_CORRECT_HORIZ,
        )
        for camera, rotation in zip(cameras, rotations):
            camera.R = rotation
        focal_lengths = sorted(float(camera.focal) for camera in cameras)
        warped_scale = float(np.median(focal_lengths))
        if not math.isfinite(warped_scale) or warped_scale <= 1:
            raise PanoramaError(
                "Invalid camera calibration",
                "The estimated camera focal length is not usable.",
                ("Use uncropped photos from the same camera and zoom setting.",),
            )

        progress(39, "Warping a cylindrical seam preview…")
        seam_warper = cv2.PyRotationWarper("cylindrical", warped_scale * seam_work_aspect)
        seam_corners: list[tuple[int, int]] = []
        seam_sizes: list[tuple[int, int]] = []
        warped_seam_images: list[np.ndarray] = []
        warped_seam_masks: list[np.ndarray] = []
        for index, image in enumerate(seam_images):
            intrinsic = cameras[index].K().astype(np.float32)
            intrinsic[0, 0] *= seam_work_aspect
            intrinsic[0, 2] *= seam_work_aspect
            intrinsic[1, 1] *= seam_work_aspect
            intrinsic[1, 2] *= seam_work_aspect
            corner, warped = seam_warper.warp(
                image,
                intrinsic,
                cameras[index].R,
                cv2.INTER_LINEAR,
                cv2.BORDER_REFLECT,
            )
            source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            _mask_corner, warped_mask = seam_warper.warp(
                source_mask,
                intrinsic,
                cameras[index].R,
                cv2.INTER_NEAREST,
                cv2.BORDER_CONSTANT,
            )
            seam_corners.append(corner)
            seam_sizes.append((warped.shape[1], warped.shape[0]))
            warped_seam_images.append(warped)
            warped_seam_masks.append(self._array(warped_mask))
            progress(
                40 + round(13 * (index + 1) / len(paths)),
                f"Preparing seamless overlap {index + 1} of {len(paths)}…",
            )

        compensator = self._exposure_compensator()
        compensator.feed(
            corners=seam_corners,
            images=warped_seam_images,
            masks=warped_seam_masks,
        )
        progress(55, "Choosing low-visibility seam paths…")
        try:
            seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
            seam_masks = seam_finder.find(
                [image.astype(np.float32) for image in warped_seam_images],
                seam_corners,
                warped_seam_masks,
            )
        except cv2.error:
            # Dynamic programming is a reliable low-memory fallback for very
            # large or unusually shaped seam canvases.
            seam_finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
            seam_masks = seam_finder.find(
                [image.astype(np.float32) for image in warped_seam_images],
                seam_corners,
                warped_seam_masks,
            )
        seam_masks = [self._array(mask) for mask in seam_masks]

        progress(61, "Compositing the full-resolution panorama…")
        compose_scale = self._megapixel_scale(reference_area, self.compose_megapixels)
        compose_work_aspect = compose_scale / work_scale
        compose_warp_scale = warped_scale * compose_work_aspect
        warper = cv2.PyRotationWarper("cylindrical", compose_warp_scale)
        compose_corners: list[tuple[int, int]] = []
        compose_sizes: list[tuple[int, int]] = []
        for index, (width, height) in enumerate(full_sizes):
            camera = cameras[index]
            camera.focal *= compose_work_aspect
            camera.ppx *= compose_work_aspect
            camera.ppy *= compose_work_aspect
            scaled_size = (max(1, round(width * compose_scale)), max(1, round(height * compose_scale)))
            roi = warper.warpRoi(scaled_size, camera.K().astype(np.float32), camera.R)
            compose_corners.append((int(roi[0]), int(roi[1])))
            compose_sizes.append((int(roi[2]), int(roi[3])))

        destination_roi = self._result_roi(compose_corners, compose_sizes)
        blend_width = math.sqrt(destination_roi[2] * destination_roi[3]) * self.blend_strength / 100.0
        blender = cv2.detail_MultiBandBlender()
        bands = max(1, min(8, int(math.log(max(2.0, blend_width), 2.0) - 1.0)))
        blender.setNumBands(bands)
        blender.prepare(destination_roi)

        for index, full_image in enumerate(full_images):
            image = self._resize(full_image, compose_scale)
            intrinsic = cameras[index].K().astype(np.float32)
            corner, warped = warper.warp(
                image,
                intrinsic,
                cameras[index].R,
                cv2.INTER_LINEAR,
                cv2.BORDER_REFLECT,
            )
            source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            _mask_corner, warped_mask = warper.warp(
                source_mask,
                intrinsic,
                cameras[index].R,
                cv2.INTER_NEAREST,
                cv2.BORDER_CONSTANT,
            )
            warped_mask = self._array(warped_mask)
            compensator.apply(index, compose_corners[index], warped, warped_mask)

            expanded_seam = cv2.dilate(seam_masks[index], None)
            expanded_seam = cv2.resize(
                expanded_seam,
                (warped_mask.shape[1], warped_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            final_mask = cv2.bitwise_and(warped_mask, expanded_seam)
            blender.feed(cv2.UMat(warped.astype(np.int16)), final_mask, corner)
            progress(
                63 + round(27 * (index + 1) / len(paths)),
                f"Blending photo {index + 1} of {len(paths)}…",
            )

        result, result_mask = blender.blend(None, None)
        result = np.clip(self._array(result), 0, 255).astype(np.uint8)
        result_mask = self._array(result_mask).astype(np.uint8)
        progress(92, "Closing the first-to-last seam and cropping empty areas…")
        cylinder, field_of_view, warnings = self._finalize_cylinder(
            result,
            result_mask,
            compose_warp_scale,
        )
        rgb = cv2.cvtColor(cylinder, cv2.COLOR_BGR2RGB)
        progress(100, "Panorama rendering complete")
        return np.ascontiguousarray(rgb), field_of_view, warnings

    @staticmethod
    def _megapixel_scale(pixel_area: int, megapixels: float) -> float:
        if megapixels < 0:
            return 1.0
        return min(1.0, math.sqrt(megapixels * 1_000_000.0 / max(1, pixel_area)))

    @staticmethod
    def _resize(image: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-3:
            return image
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

    @staticmethod
    def _circular_matching_mask(count: int) -> np.ndarray:
        mask = np.zeros((count, count), dtype=np.uint8)
        for index in range(count):
            neighbor = (index + 1) % count
            mask[index, neighbor] = 1
            mask[neighbor, index] = 1
            # Including second neighbors stabilizes focal estimation when the
            # source sequence contains generous overlap.
            if count >= 6:
                second = (index + 2) % count
                mask[index, second] = 1
                mask[second, index] = 1
        return mask

    @staticmethod
    def _exposure_compensator():
        try:
            return cv2.detail_BlocksChannelsCompensator(32, 32, 2)
        except (AttributeError, TypeError, cv2.error):
            return cv2.detail.ExposureCompensator_createDefault(
                cv2.detail.ExposureCompensator_GAIN_BLOCKS
            )

    @staticmethod
    def _array(value):
        return value.get() if hasattr(value, "get") else np.asarray(value)

    @staticmethod
    def _result_roi(
        corners: list[tuple[int, int]], sizes: list[tuple[int, int]]
    ) -> tuple[int, int, int, int]:
        try:
            roi = cv2.detail.resultRoi(corners=corners, sizes=sizes)
        except TypeError:
            roi = cv2.detail.resultRoi(corners, sizes)
        return tuple(map(int, roi))

    @staticmethod
    def _blend_duplicate_column(
        output: np.ndarray,
        output_mask: np.ndarray,
        destination_x: int,
        source_column: np.ndarray,
        source_mask: np.ndarray,
    ) -> None:
        valid_source = source_mask > 0
        valid_output = output_mask[:, destination_x] > 0
        only_source = valid_source & ~valid_output
        both = valid_source & valid_output
        output[only_source, destination_x] = source_column[only_source]
        if np.any(both):
            mixed = (
                output[both, destination_x].astype(np.uint16)
                + source_column[both].astype(np.uint16)
            ) // 2
            output[both, destination_x] = mixed.astype(np.uint8)
        output_mask[valid_source, destination_x] = 255

    def _finalize_cylinder(
        self,
        canvas: np.ndarray,
        canvas_mask: np.ndarray,
        pixels_per_radian: float,
    ) -> tuple[np.ndarray, float, list[str]]:
        warnings: list[str] = []
        expected_width = max(1, round(2.0 * math.pi * pixels_per_radian))
        canvas_width = canvas.shape[1]
        coverage_degrees = min(360.0, 360.0 * canvas_width / expected_width)
        if canvas_width < expected_width * 0.90:
            raise PanoramaError(
                "Incomplete horizontal coverage",
                f"Camera calibration found only about {coverage_degrees:.0f}° of usable horizontal coverage.",
                (
                    "Include the missing frames from the rotation.",
                    "Make sure the last photo overlaps the first by at least 25%.",
                ),
            )

        # Small focal-estimation drift is expected in a closed loop. When the
        # calibrated circumference exceeds the occupied canvas by less than 10%,
        # the loop closure is stronger evidence than the raw focal estimate.
        cylinder_width = min(expected_width, canvas_width)
        start = max(0, (canvas_width - cylinder_width) // 2)
        end = start + cylinder_width
        output = canvas[:, start:end].copy()
        output_mask = canvas_mask[:, start:end].copy()

        # Warped images that cross ±180° appear on both sides of OpenCV's canvas.
        # Fold those duplicate columns back onto the cylindrical interval. This
        # blends observations of the same direction and makes the array periodic.
        for source_x in range(0, start):
            destination_x = (source_x - start) % cylinder_width
            self._blend_duplicate_column(
                output,
                output_mask,
                destination_x,
                canvas[:, source_x],
                canvas_mask[:, source_x],
            )
        for source_x in range(end, canvas_width):
            destination_x = (source_x - start) % cylinder_width
            self._blend_duplicate_column(
                output,
                output_mask,
                destination_x,
                canvas[:, source_x],
                canvas_mask[:, source_x],
            )

        valid_fraction = (output_mask > 0).mean(axis=1)
        fully_covered = valid_fraction >= 0.995
        top, bottom = self._longest_true_run(fully_covered)
        if bottom - top < max(64, round(output.shape[0] * 0.18)):
            raise PanoramaError(
                "No clean rectangular panorama area",
                "Alignment left black or empty gaps across too much of the image.",
                (
                    "Keep the camera level and rotate around one fixed point.",
                    "Use more vertical overlap and avoid mixing portrait and landscape photos.",
                ),
            )
        output = output[top:bottom]
        output_mask = output_mask[top:bottom]
        hole_fraction = float((output_mask == 0).mean())
        if hole_fraction > 0.005:
            raise PanoramaError(
                "Unfilled areas remain",
                "The aligned photos leave visible empty regions inside the panorama.",
                ("Retake the rotation with steadier framing and more overlap.",),
            )
        if hole_fraction > 0:
            # These are sub-pixel boundary pinholes, not missing scene content.
            holes = np.where(output_mask == 0, 255, 0).astype(np.uint8)
            output = cv2.inpaint(output, holes, 2, cv2.INPAINT_TELEA)

        seam_difference = float(
            np.mean(
                np.abs(
                    output[:, 0].astype(np.int16) - output[:, -1].astype(np.int16)
                )
            )
        )
        adjacent_difference = float(
            np.median(
                np.mean(
                    np.abs(output[:, 1:].astype(np.int16) - output[:, :-1].astype(np.int16)),
                    axis=(0, 2),
                )
            )
        )
        if seam_difference > max(28.0, adjacent_difference * 5.0):
            warnings.append(
                "The wrap boundary contains a high-contrast subject; inspect the first-to-last seam at 100% zoom."
            )

        return np.ascontiguousarray(output), 360.0, warnings

    @staticmethod
    def _longest_true_run(values: np.ndarray) -> tuple[int, int]:
        best_start = best_end = run_start = 0
        in_run = False
        for index, value in enumerate(values):
            if value and not in_run:
                run_start = index
                in_run = True
            if in_run and (not value or index == len(values) - 1):
                run_end = index + 1 if value and index == len(values) - 1 else index
                if run_end - run_start > best_end - best_start:
                    best_start, best_end = run_start, run_end
                in_run = False
        return best_start, best_end
