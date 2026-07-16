"""Image loading and saving with EXIF orientation and Unicode paths."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .models import PanoramaError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as a contiguous 8-bit RGB array."""

    source = Path(path)
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
    except Exception as exc:
        raise PanoramaError(
            "Could not open photo",
            f"{source.name} could not be read as an image.\n\n{exc}",
            ("Choose an unmodified JPEG, PNG, TIFF, BMP, or WebP file.",),
        ) from exc

    if array.shape[0] < 240 or array.shape[1] < 320:
        raise PanoramaError(
            "Photo is too small",
            f"{source.name} is only {array.shape[1]} × {array.shape[0]} pixels.",
            ("Use the original camera photos rather than thumbnails.",),
        )
    return np.ascontiguousarray(array)


def save_rgb(image_rgb: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """Atomically save an RGB panorama, choosing format from the suffix."""

    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        destination = destination.with_suffix(".jpg")
        suffix = ".jpg"

    temporary = destination.with_name(f".{destination.stem}.saving{destination.suffix}")
    image = Image.fromarray(np.ascontiguousarray(image_rgb), mode="RGB")
    options: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg", ".webp"}:
        options.update(quality=quality, optimize=True)
    if suffix in {".jpg", ".jpeg"}:
        options["subsampling"] = 0
    if suffix in {".tif", ".tiff"}:
        options["compression"] = "tiff_lzw"

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(temporary, **options)
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise PanoramaError(
            "Could not save panorama",
            f"The panorama could not be saved to {destination}.\n\n{exc}",
            ("Choose a writable folder and try again.",),
        ) from exc
