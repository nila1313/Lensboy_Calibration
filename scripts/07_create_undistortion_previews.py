from pathlib import Path
import json
import cv2
import yaml
import numpy as np


def main():
    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    output_folder = Path(paths_config["single_output_folder"])

    frames_folder = output_folder / "frames"
    reports_folder = output_folder / "reports"
    preview_folder = output_folder / "undistorted" / "frames"
    side_by_side_folder = output_folder / "undistorted" / "side_by_side"

    preview_folder.mkdir(parents=True, exist_ok=True)
    side_by_side_folder.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_folder / "lensboy_calibration_metrics.json"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    K = np.asarray(metrics["K"], dtype=np.float64)
    dist = np.asarray(metrics["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

    frame_paths = sorted(frames_folder.glob("*.png"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_folder}")

    # Choose frames across the whole video.
    n_samples = min(20, len(frame_paths))
    indices = np.linspace(0, len(frame_paths) - 1, n_samples).round().astype(int)

    print("=" * 80)
    print("UNDISTORTION PREVIEW")
    print("=" * 80)
    print(f"Using model metrics: {metrics_path}")
    print(f"Input frames: {frames_folder}")
    print(f"Undistorted frames output: {preview_folder}")
    print(f"Side-by-side output: {side_by_side_folder}")
    print()

    saved = 0

    for idx in indices:
        frame_path = frame_paths[idx]
        img = cv2.imread(str(frame_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K,
            dist,
            (w, h),
            alpha=1.0,
            newImgSize=(w, h)
        )

        undistorted = cv2.undistort(img, K, dist, None, new_K)

        # Add text labels
        original_labeled = img.copy()
        undistorted_labeled = undistorted.copy()

        cv2.putText(
            original_labeled,
            "Original",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4,
            cv2.LINE_AA
        )

        cv2.putText(
            undistorted_labeled,
            "Undistorted",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 180, 0),
            4,
            cv2.LINE_AA
        )

        side_by_side = np.hstack([original_labeled, undistorted_labeled])

        undistorted_path = preview_folder / frame_path.name
        side_by_side_path = side_by_side_folder / frame_path.name

        cv2.imwrite(str(undistorted_path), undistorted)
        cv2.imwrite(str(side_by_side_path), side_by_side)

        saved += 1

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved preview frames: {saved}")
    print(f"Open side-by-side folder:")
    print(side_by_side_folder)


if __name__ == "__main__":
    main()
