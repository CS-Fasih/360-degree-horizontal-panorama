from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from panorama360.stitcher import CylindricalStitcher


def test_circular_matching_mask_includes_first_last_and_second_neighbors() -> None:
    mask = CylindricalStitcher._circular_matching_mask(8)
    assert np.array_equal(mask, mask.T)
    assert mask[0, 7] == 1
    assert mask[0, 6] == 1
    assert mask[0, 3] == 0
    assert not np.any(np.diag(mask))


def test_finalize_cylinder_folds_duplicate_wrap_and_crops_vertical_empty_area() -> None:
    stitcher = CylindricalStitcher()
    circumference = 200
    scale = circumference / (2 * math.pi)
    height = 90
    extra = 20
    canvas = np.zeros((height, circumference + 2 * extra, 3), dtype=np.uint8)
    mask = np.zeros((height, circumference + 2 * extra), dtype=np.uint8)
    # A valid cylindrical band with duplicate pixels on both sides.
    gradient = np.linspace(10, 240, circumference, dtype=np.uint8)
    canvas[12:78, extra : extra + circumference, 0] = gradient
    canvas[12:78, extra : extra + circumference, 1] = 100
    canvas[12:78, extra : extra + circumference, 2] = gradient[::-1]
    mask[12:78, extra : extra + circumference] = 255
    canvas[12:78, :extra] = canvas[12:78, circumference : circumference + extra]
    mask[12:78, :extra] = 255
    canvas[12:78, extra + circumference :] = canvas[12:78, extra : 2 * extra]
    mask[12:78, extra + circumference :] = 255

    result, field_of_view, warnings = stitcher._finalize_cylinder(canvas, mask, scale)

    assert result.shape == (66, circumference, 3)
    assert field_of_view == 360.0
    assert isinstance(warnings, list)
    assert not np.any(np.all(result == 0, axis=2))


def test_longest_true_run_handles_run_at_array_end() -> None:
    values = np.array([False, True, True, False, True, True, True], dtype=bool)
    assert CylindricalStitcher._longest_true_run(values) == (4, 7)


def test_end_to_end_spherical_camera_loop_renders_cylindrical_360(tmp_path: Path) -> None:
    """Exercise native calibration, warping, seams, exposure, and blending."""

    rng = np.random.default_rng(360)
    world_height, world_width = 520, 1800
    world = np.full((world_height, world_width, 3), 160, dtype=np.uint8)
    world[..., 0] = np.linspace(35, 225, world_height, dtype=np.uint8)[:, None]
    world[..., 1] = np.linspace(190, 75, world_height, dtype=np.uint8)[:, None]
    for _ in range(1700):
        center = (int(rng.integers(world_width)), int(rng.integers(world_height)))
        radius = int(rng.integers(2, 9))
        color = tuple(int(value) for value in rng.integers(5, 250, 3))
        cv2.circle(world, center, radius, color, -1, cv2.LINE_AA)

    count = 6
    image_height, image_width = 300, 420
    focal = (image_width / 2) / math.tan(math.radians(92 / 2))
    x_axis = (np.arange(image_width) - image_width / 2) / focal
    y_axis = (np.arange(image_height) - image_height / 2) / focal
    ray_x, ray_y = np.meshgrid(x_axis, y_axis)
    ray_z = np.ones_like(ray_x)
    paths: list[Path] = []

    for index in range(count):
        yaw = 2 * math.pi * index / count
        cosine, sine = math.cos(yaw), math.sin(yaw)
        world_x = cosine * ray_x + sine * ray_z
        world_z = -sine * ray_x + cosine * ray_z
        longitude = np.arctan2(world_x, world_z)
        latitude = np.arctan2(ray_y, np.sqrt(world_x**2 + world_z**2))
        map_x = ((longitude / (2 * math.pi) + 0.5) * world_width).astype(np.float32)
        map_y = ((latitude / math.pi + 0.5) * world_height).astype(np.float32)
        view = cv2.remap(world, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        path = tmp_path / f"view_{index}.jpg"
        Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)).save(path, quality=96)
        paths.append(path)

    result = CylindricalStitcher(
        work_megapixels=0.12,
        seam_megapixels=0.04,
        compose_megapixels=0.12,
    ).create(paths, auto_order=False)

    assert result.field_of_view_degrees == 360.0
    assert result.output_width > result.output_height * 4
    assert result.output_height >= 180
    assert not np.any(np.all(result.image_rgb == 0, axis=2))
