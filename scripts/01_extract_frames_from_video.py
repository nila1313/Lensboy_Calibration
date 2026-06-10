from pathlib import Path
import argparse
import csv
import json
import cv2
import yaml


def find_video(input_folder: Path):
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    videos = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    if len(videos) == 0:
        raise FileNotFoundError(f"No video found in {input_folder}")

    if len(videos) > 1:
        print("More than one video found. Using the first one:")
        for v in videos:
            print("  ", v)

    return videos[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=30,
        help="Extract one frame every N video frames. Default: 30"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to extract."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete old extracted frames before writing new ones."
    )
    args = parser.parse_args()

    config_path = Path("configs/paths.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_folder = Path(config["single_input_folder"])
    output_folder = Path(config["single_output_folder"])

    frames_folder = output_folder / "frames"
    reports_folder = output_folder / "reports"
    logs_folder = output_folder / "logs"

    frames_folder.mkdir(parents=True, exist_ok=True)
    reports_folder.mkdir(parents=True, exist_ok=True)
    logs_folder.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        old_frames = list(frames_folder.glob("*.png"))
        for frame_file in old_frames:
            frame_file.unlink()
        print(f"Deleted old frames: {len(old_frames)}")

    video_path = find_video(input_folder)

    print("=" * 80)
    print("FRAME EXTRACTION")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Output frames folder: {frames_folder}")
    print(f"Extract every N frames: {args.every_n_frames}")
    print()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = total_frames / fps if fps > 0 else None

    video_info = {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "every_n_frames": args.every_n_frames,
        "max_frames": args.max_frames,
    }

    with open(reports_folder / "video_info.json", "w") as f:
        json.dump(video_info, f, indent=4)

    manifest_path = reports_folder / "frame_manifest.csv"

    saved_count = 0
    frame_index = 0

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "saved_frame_name",
            "original_video_frame_index",
            "timestamp_seconds"
        ])

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % args.every_n_frames == 0:
                timestamp_seconds = frame_index / fps if fps > 0 else None
                frame_name = f"frame_{frame_index:06d}.png"
                frame_path = frames_folder / frame_name

                success = cv2.imwrite(str(frame_path), frame)

                if not success:
                    raise RuntimeError(f"Could not write frame: {frame_path}")

                writer.writerow([
                    frame_name,
                    frame_index,
                    timestamp_seconds
                ])

                saved_count += 1

                if saved_count % 20 == 0:
                    print(f"Saved {saved_count} frames...")

                if args.max_frames is not None and saved_count >= args.max_frames:
                    break

            frame_index += 1

    cap.release()

    extraction_summary = {
        "video_path": str(video_path),
        "frames_folder": str(frames_folder),
        "saved_frames": saved_count,
        "every_n_frames": args.every_n_frames,
        "total_video_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }

    with open(reports_folder / "extraction_summary.json", "w") as f:
        json.dump(extraction_summary, f, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved frames: {saved_count}")
    print(f"Frames folder: {frames_folder}")
    print(f"Manifest: {manifest_path}")
    print(f"Video info: {reports_folder / 'video_info.json'}")


if __name__ == "__main__":
    main()
