# Panorama 360

Panorama 360 is a local Python desktop application that turns one complete rotation of overlapping photos into a seamless horizontal cylindrical panorama. It accepts any reasonable photo count—the sequence is discovered from image content, not from filenames or a fixed 6/12/18-image template.

The program never invents missing sky or ground. If the photos cannot support a dependable 360° result, it explains the problem instead of exporting a visibly broken image.

## What it does

- Loads JPEG, PNG, TIFF, BMP, and WebP photos with EXIF orientation applied.
- Shows draggable thumbnails and automatically finds a circular photo order.
- Uses SIFT features, bidirectional matching, robust homographies, and a circular graph search to identify real overlap.
- Verifies every adjacent pair and explicitly verifies the last-to-first overlap.
- Solves all camera rotations together with bundle adjustment.
- Straightens the horizon and uses a true cylindrical projection.
- Compensates block-level exposure and colour differences.
- Finds low-visibility seams and uses multiband blending to suppress them.
- Folds duplicate ±180° observations together so the saved image wraps correctly.
- Crops to a fully covered rectangular band without generating missing content.
- Reports real analysis/rendering progress in a responsive GUI.
- Provides a zoomable preview and saves full-quality JPEG, PNG, TIFF, or WebP output.
- Keeps soft-but-matchable photos as warnings and retries matching/calibration with safer fallback profiles instead of failing on blur alone.

## Install and run

Python 3.10–3.14 is supported. A virtual environment is strongly recommended.

```bash
git clone https://github.com/CS-Fasih/360-degree-horizontal-panorama.git
cd 360-degree-horizontal-panorama
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python main.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
python -m pip install -e .
python main.py
```

After installation, the `panorama360` command is also available.

### Linux package note

PySide6 wheels include Qt, but some minimal Linux installations still need common desktop libraries. If Qt reports a missing platform dependency on Ubuntu/Debian, install:

```bash
sudo apt install libegl1 libgl1 libxkbcommon-x11-0
```

## Workflow

1. Click **Add photos…** and select all original photos at once with Ctrl/Shift (or use **Add folder…** to import a whole folder) from one continuous horizontal rotation.
2. Wait while the application analyzes overlap and automatically reorders the thumbnails.
3. Inspect the order. Drag thumbnails if a manual correction is needed; the last item must overlap the first.
4. Click **Create Panorama**. The displayed order is verified again before expensive rendering begins.
5. Inspect the result, including the horizontal wrap boundary, and click **Save panorama…**.

## Capture guidance

Good source photography matters more than any stitching setting:

- Stand in one place and rotate the camera rather than stepping sideways.
- Keep roughly 25–50% repeated content between neighboring photos.
- Complete the full turn and overlap the final photo with the first.
- Lock focus, exposure, and white balance when the camera allows it.
- Keep the camera level and avoid zooming during the sequence.
- Prefer a static scene. Moving people, vehicles, water, and nearby leaves can create ghosting.
- For nearby scenes, rotate around the lens entrance pupil with a panoramic head to minimize parallax.
- Use original files rather than resized social-media copies or thumbnails.

Six photos at roughly 60° intervals can work with a sufficiently wide field of view. Twelve at 30° or eighteen at 20° generally provide more overlap. The exact count is not important; dependable overlap and fixed-position rotation are.

## When the application refuses a set

Rendering is stopped when analysis finds conditions such as:

- no strong overlap between one or more neighboring photos;
- no dependable first-to-last closure, which means 360° coverage is unproven;
- too few recognizable features or severe blur;
- incompatible photo orientations/aspect ratios;
- camera motion or parallax too inconsistent for global calibration;
- black or unfilled regions remaining after cylindrical projection.

Warnings are shown for softer photos, large exposure changes, vertical camera drift, possible parallax, and a high-contrast wrap boundary. Manual ordering can correct a sequencing mistake, but it cannot repair missing coverage or camera translation.

## Technical pipeline

```text
EXIF-correct RGB load
  → quality and feature diagnostics
  → all-pairs bidirectional SIFT matching
  → robust homography/inlier validation
  → maximum-confidence circular order
  → constrained closed-loop matching
  → camera estimation and global bundle adjustment
  → horizontal wave correction
  → cylindrical warping
  → block colour/exposure compensation
  → graph-cut seam selection
  → multiband full-resolution blending
  → periodic wrap folding
  → valid-band crop and output validation
```

Feature registration and seam discovery use reduced working copies for speed. Final composition uses up to 12 megapixels from each source photo by default, producing a high-resolution panorama while keeping memory use practical on a typical local workstation.

## Development and tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The suite includes circular graph ordering, rejection of missing closure, real SIFT analysis of scrambled periodic photos, wrap folding/cropping, image I/O, and an end-to-end six-camera synthetic spherical rotation through OpenCV calibration, seams, and blending.

## Important expectation

No stitching program can guarantee a flawless result from every input. Missing coverage, strong parallax, motion blur, repeated texture, or moving subjects destroy information needed for correct alignment. This project prioritizes a genuine cylindrical 360° output and refuses unreliable sets rather than fabricating pixels or silently exporting a broken panorama.
