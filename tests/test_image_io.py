from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from panorama360.image_io import load_rgb, save_rgb


def test_rgb_round_trip_png(tmp_path: Path) -> None:
    source = np.zeros((260, 340, 3), dtype=np.uint8)
    source[..., 0] = 31
    source[..., 1] = 127
    source[..., 2] = 229
    path = tmp_path / "result.png"

    save_rgb(source, path)
    loaded = load_rgb(path)

    assert np.array_equal(loaded, source)


def test_unknown_save_suffix_defaults_to_jpeg(tmp_path: Path) -> None:
    source = np.full((260, 340, 3), 150, dtype=np.uint8)
    requested = tmp_path / "result.unknown"
    save_rgb(source, requested)
    assert (tmp_path / "result.jpg").exists()
