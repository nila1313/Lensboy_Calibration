# Lensboy Calibration

Camera calibration workspace for Lensboy experiments with ChArUco boards. The repository contains two calibration flows:

- Single-camera calibration for the `blur-distortion-overexposure` dataset.
- Dark multicamera calibration for the `dark_frames-no_common_pose_frame` dataset.

The scripts are intentionally numbered so they can be run as a pipeline, inspected one step at a time, and rerun from the point where data or board settings change.

## Repository Layout

```text
Lensboy_Calibration/
├── configs/
│   ├── paths.yaml                         # Single-camera input and output paths
│   ├── charuco_board.yaml                 # Single-camera ChArUco board settings
│   ├── dark_multicam_paths.yaml           # Dark multicamera input and output paths
│   └── dark_multicam_charuco_board.yaml   # Dark multicamera ChArUco board settings
├── scripts/
│   ├── 00-08_*.py                         # Single-camera inspection, detection, calibration, output
│   ├── 10-15b_*.py                        # Dark multicamera inspection, detection, poses, relative extrinsics
│   └── 12c-12e_*.py                       # Dictionary and board-layout diagnostics
└── outputs/
    ├── single/...                         # Example single-camera reports and plots
    └── multicamera/...                    # Dark multicamera reports and plots
```

## Requirements

Use a Python environment with:

- `opencv-contrib-python`
- `numpy`
- `PyYAML`
- `matplotlib`
- `lensboy`

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-contrib-python numpy PyYAML matplotlib lensboy
```

## Configuration

Before running scripts, update the paths in:

- `configs/paths.yaml` for the single-camera dataset.
- `configs/dark_multicam_paths.yaml` for the dark multicamera dataset.

Board geometry is stored separately:

- `configs/charuco_board.yaml`
- `configs/dark_multicam_charuco_board.yaml`

The dark multicamera board currently uses `DICT_4X4_50`, a `6 x 8` board, `3.0` square size, `1.8` marker size, and `legacy_pattern: true`.

## Single-Camera Pipeline

Run from the repository root:

```bash
python scripts/00_inspect_input_folder.py
python scripts/01_extract_frames_from_video.py
python scripts/02_inspect_board_npy.py
python scripts/03_detect_charuco.py
python scripts/04_calibrate_single_camera_lensboy.py
python scripts/05_evaluate_bad_frames.py
python scripts/06_make_clean_detections.py
python scripts/05b_evaluate_clean_frames.py
python scripts/07_create_undistortion_previews.py
python scripts/07b_create_undistortion_alpha_comparison.py
python scripts/08_create_final_undistorted_video.py
```

Typical outputs include detection summaries, calibration metrics, per-frame reprojection errors, diagnostic plots, undistorted preview frames, and an undistorted video.

## Dark Multicamera Pipeline

Run from the repository root:

```bash
python scripts/10_inspect_dark_multicam_dataset.py
python scripts/11_extract_dark_multicam_frames.py
python scripts/12_detect_dark_multicam_charuco.py
python scripts/13_calibrate_dark_multicam_intrinsics.py
python scripts/14_estimate_dark_multicam_board_poses.py
python scripts/15_estimate_dark_multicam_relative_poses.py
python scripts/15b_estimate_dark_multicam_relative_poses_robust.py
```

Useful diagnostic scripts:

```bash
python scripts/12c_test_aruco_dictionaries_dark_multicam.py
python scripts/12d_test_charuco_dictionaries_dark_multicam.py
python scripts/12e_test_board_layout_variants_dark_multicam.py
```

The robust relative-pose estimator in `15b` is the preferred final multicamera extrinsics step when noisy frames or missing shared poses make the direct estimate unstable.

## Outputs

Generated files are written under the configured output folders. The main report-style artifacts are usually in:

- `reports/` for JSON, CSV, and text summaries.
- `plots/` for calibration and reprojection diagnostics.
- `frames/` for extracted frames.
- `undistorted/` for preview images and videos.

The repository keeps lightweight example outputs, but large generated datasets and videos should stay outside Git unless they are intentionally part of a result snapshot.

Tracked multicamera outputs include:

- Per-camera Lensboy calibration reports and diagnostic plots under `outputs/multicamera/dark_frames-no_common_pose_frame/cam*/reports/` and `cam*/plots/`.
- Rig-level pose reports under `outputs/multicamera/dark_frames-no_common_pose_frame/reports/`.
- Rig-level camera layout plots under `outputs/multicamera/dark_frames-no_common_pose_frame/plots/`.

Extracted multicamera frames, videos, and other heavy generated artifacts are intentionally ignored.

## Notes

- Run commands from the repository root so relative config paths resolve correctly.
- If OpenCV ChArUco behavior changes between versions, the scripts include compatibility handling for both older and newer `cv2.aruco` APIs.
- Keep the YAML configs as the source of truth for data locations and board geometry before rerunning a pipeline.
