"""Shared data structures for analysis, stitching, and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


ProgressCallback = Callable[[int, str], None]


class PanoramaError(RuntimeError):
    """A recoverable, user-facing panorama failure."""

    def __init__(
        self,
        title: str,
        detail: str,
        suggestions: tuple[str, ...] = (),
    ) -> None:
        super().__init__(detail)
        self.title = title
        self.detail = detail
        self.suggestions = suggestions

    def user_message(self) -> str:
        message = self.detail
        if self.suggestions:
            message += "\n\nTry:\n" + "\n".join(f"• {item}" for item in self.suggestions)
        return message


@dataclass(slots=True)
class ImageQuality:
    """Basic diagnostics calculated on an analysis-sized image."""

    width: int
    height: int
    blur_score: float
    feature_count: int
    mean_luminance: float


@dataclass(slots=True)
class PairwiseMatch:
    """Geometric match from image ``left`` into image ``right``."""

    left: int
    right: int
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    source_coverage: float = 0.0
    target_coverage: float = 0.0
    median_error: float = float("inf")
    horizontal_shift: float = 0.0
    vertical_shift: float = 0.0
    homography: np.ndarray | None = field(default=None, repr=False)

    @property
    def coverage(self) -> float:
        return min(self.source_coverage, self.target_coverage)

    @property
    def is_usable(self) -> bool:
        return (
            self.homography is not None
            and self.inliers >= 14
            and self.inlier_ratio >= 0.16
            and self.coverage >= 0.008
            and self.median_error <= 6.5
        )

    @property
    def score(self) -> float:
        if self.homography is None:
            return 0.0
        # Confidence needs both geometric agreement and spatially distributed
        # features. A cluster of matches on one small object is not enough.
        spatial = 0.35 + min(0.65, self.coverage * 5.0)
        precision = max(0.0, 1.0 - self.median_error / 12.0)
        return float(self.inliers * self.inlier_ratio * spatial * precision)


@dataclass(slots=True)
class AnalysisResult:
    """Result of the inexpensive preflight analysis."""

    paths: list[Path]
    order: list[int]
    matches: dict[tuple[int, int], PairwiseMatch]
    qualities: list[ImageQuality]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    has_circular_closure: bool = False

    @property
    def ordered_paths(self) -> list[Path]:
        return [self.paths[index] for index in self.order]

    @property
    def can_stitch(self) -> bool:
        return len(self.paths) >= 3 and not self.errors and self.has_circular_closure

    def match_between(self, a: int, b: int) -> PairwiseMatch | None:
        return self.matches.get((min(a, b), max(a, b)))


@dataclass(slots=True)
class StitchResult:
    """Completed panorama and its diagnostics."""

    image_rgb: np.ndarray
    analysis: AnalysisResult
    output_width: int
    output_height: int
    field_of_view_degrees: float
    warnings: list[str] = field(default_factory=list)
