from pathlib import Path
import json
import cv2
import yaml
import numpy as np


def resize_for_preview(img, width=480):
    h, w = img.shape[:2]
    new_h = int(width * h / w)
    return cv2.resize(img, (width, new_h))


def put_label(img, text, color):
    out = img.copy()
    cv2.putText(
        out,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA
    )
    return out


def main():
    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    output_folder = Path(paths_config["single_output_folder"])

    frames_folder = output_folder / "frames"
    reports_folder = output_folder / "reports"
    comparison_folder = output_folder / "undistorted" / "alpha_comparison"

    comparison_folder.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_folder / "lensboy_calibration_metrics.json"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    K = np.asarray(metrics["K"], dtype=np.float64)
    dist = np.asarray(metrics["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

    frame_paths = sorted(frames_folder.glob("*.png"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_folder}")

    # Pick 10 frames across the video.
    n_samples = min(10, len(frame_paths))
    indices = np.linspace(0, len(frame_paths) - 1, n_samples).round().astype(int)

    alphas = [0.0, 0.3, 0.6, 1.0]

    print("=" * 80)
    print("UNDISTORTION ALPHA COMPARISON")
    print("=" * 80)
    print(f"Input frames: {frames_folder}")
    print(f"Output folder: {comparison_folder}")
    print(f"Alphas: {alphas}")
    print()

    saved = 0

    for idx in indices:
        frame_path = frame_paths[idx]
        img = cv2.imread(str(frame_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        preview_tiles = []

        original_preview = resize_for_preview(img)
        original_preview = put_label(original_preview, "Original", (0, 0, 255))
        preview_tiles.append(original_preview)

        for alpha in alphas:
            new_K, roi = cv2.getOptimalNewCameraMatrix(
                K,
                dist,
                (w, h),
                alpha=alpha,
                newImgSize=(w, h)
            )

            undistorted = cv2.undistort(img, K, dist, None, new_K)

            # For alpha=0, crop to ROI for cleaner view.
            if alpha == 0.0:
                x, y, rw, rh = roi
                if rw > 0 and rh > 0:
                    undistorted = undistorted[y:y+rh, x:x+rw]

            undistorted_preview = resize_for_preview(undistorted)
            undistorted_preview = put_label(
                undistorted_preview,
                f"alpha={alpha}",
                (0, 180, 0)
            )

            preview_tiles.append(undistorted_preview)

        # Make all tiles same height by padding.
        max_h = max(tile.shape[0] for tile in preview_tiles)
        padded_tiles = []

        for tile in preview_tiles:
            th, tw = tile.shape[:2]
            if th < max_h:
                pad = np.zeros((max_h - th, tw, 3), dtype=np.uint8)
                tile = np.vstack([tile, pad])
            padded_tiles.append(tile)

        comparison = np.hstack(padded_tiles)

        out_path = comparison_folder / frame_path.name
        cv2.imwrite(str(out_path), comparison)
        saved += 1

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved comparisons: {saved}")
    print(f"Open this folder:")
    print(comparison_folder)


if __name__ == "__main__":
    main()
