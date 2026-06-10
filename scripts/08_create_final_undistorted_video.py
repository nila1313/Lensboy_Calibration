from pathlib import Path
import argparse
import json
import cv2
import yaml
import numpy as np


def find_video(input_folder: Path):
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    videos = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    if len(videos) == 0:
        raise FileNotFoundError(f"No video found in {input_folder}")

    return videos[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--crop-to-roi", action="store_true")
    args = parser.parse_args()

    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    input_folder = Path(paths_config["single_input_folder"])
    output_folder = Path(paths_config["single_output_folder"])

    reports_folder = output_folder / "reports"
    video_output_folder = output_folder / "undistorted" / "video"
    video_output_folder.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_folder / "lensboy_calibration_metrics.json"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    K = np.asarray(metrics["K"], dtype=np.float64)
    dist = np.asarray(metrics["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

    video_path = find_video(input_folder)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        (width, height),
        alpha=args.alpha,
        newImgSize=(width, height)
    )

    if args.crop_to_roi:
        x, y, rw, rh = roi
        if rw <= 0 or rh <= 0:
            raise RuntimeError(f"Invalid ROI: {roi}")
        output_size = (rw, rh)
    else:
        x, y, rw, rh = 0, 0, width, height
        output_size = (width, height)

    output_name = f"undistorted_alpha_{args.alpha:g}"
    if args.crop_to_roi:
        output_name += "_cropped"
    if args.max_frames is not None:
        output_name += f"_first_{args.max_frames}_frames"
    output_name += ".mp4"

    output_video_path = video_output_folder / output_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        output_size
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    print("=" * 80)
    print("FINAL UNDISTORTED VIDEO")
    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_video_path}")
    print(f"Alpha: {args.alpha}")
    print(f"Crop to ROI: {args.crop_to_roi}")
    print(f"Input size: {width} x {height}")
    print(f"Output size: {output_size[0]} x {output_size[1]}")
    print(f"FPS: {fps}")
    print(f"Total input frames: {total_frames}")
    print()

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        undistorted = cv2.undistort(frame, K, dist, None, new_K)

        if args.crop_to_roi:
            undistorted = undistorted[y:y+rh, x:x+rw]

        writer.write(undistorted)

        written += 1
        frame_idx += 1

        if written % 100 == 0:
            print(f"Written {written} frames...")

        if args.max_frames is not None and written >= args.max_frames:
            break

    cap.release()
    writer.release()

    summary = {
        "input_video": str(video_path),
        "output_video": str(output_video_path),
        "alpha": args.alpha,
        "crop_to_roi": args.crop_to_roi,
        "input_width": width,
        "input_height": height,
        "output_width": output_size[0],
        "output_height": output_size[1],
        "fps": fps,
        "input_total_frames": total_frames,
        "written_frames": written,
        "model_metrics": str(metrics_path),
    }

    summary_path = reports_folder / f"undistorted_video_alpha_{args.alpha:g}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Written frames: {written}")
    print(f"Video saved to: {output_video_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
