from pathlib import Path
import argparse
import csv
import json
import re
import cv2
import yaml


def get_cam_name(video_path: Path):
    match = re.search(r"cam(\d+)", video_path.name)
    if not match:
        raise ValueError(f"Could not find camera number in filename: {video_path.name}")
    return f"cam{match.group(1)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        help="Extract one frame every N frames. For multicamera, 1 is recommended first."
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality from 1 to 100."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete old extracted JPG frames before extracting new ones."
    )
    args = parser.parse_args()

    with open("configs/dark_multicam_paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_folder = Path(config["input_folder"])
    output_folder = Path(config["output_folder"])
    reports_folder = output_folder / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    videos = sorted([
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ])

    if len(videos) == 0:
        raise FileNotFoundError(f"No videos found in {input_folder}")

    print("=" * 80)
    print("DARK MULTICAMERA FRAME EXTRACTION")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Videos found: {len(videos)}")
    print(f"Extract every N frames: {args.every_n_frames}")
    print(f"JPEG quality: {args.jpeg_quality}")
    print()

    dataset_manifest = []

    for video_path in videos:
        cam_name = get_cam_name(video_path)
        cam_folder = output_folder / cam_name
        frames_folder = cam_folder / "frames"
        reports_cam_folder = cam_folder / "reports"

        frames_folder.mkdir(parents=True, exist_ok=True)
        reports_cam_folder.mkdir(parents=True, exist_ok=True)

        if args.overwrite:
            old_frames = list(frames_folder.glob("*.jpg"))
            for old in old_frames:
                old.unlink()
            print(f"{cam_name}: deleted old JPG frames: {len(old_frames)}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("-" * 80)
        print(f"Camera: {cam_name}")
        print(f"Video: {video_path.name}")
        print(f"Frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Size: {width} x {height}")

        manifest_path = reports_cam_folder / "frame_manifest.csv"

        saved = 0
        frame_idx = 0

        with open(manifest_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "camera",
                "saved_frame_name",
                "original_video_frame_index",
                "timestamp_seconds"
            ])

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_idx % args.every_n_frames == 0:
                    timestamp = frame_idx / fps if fps > 0 else None
                    frame_name = f"frame_{frame_idx:06d}.jpg"
                    frame_path = frames_folder / frame_name

                    ok = cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                    )

                    if not ok:
                        raise RuntimeError(f"Could not write frame: {frame_path}")

                    writer.writerow([
                        cam_name,
                        frame_name,
                        frame_idx,
                        timestamp
                    ])

                    saved += 1

                frame_idx += 1

        cap.release()

        cam_summary = {
            "camera": cam_name,
            "video_path": str(video_path),
            "total_video_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "every_n_frames": args.every_n_frames,
            "saved_frames": saved,
            "frames_folder": str(frames_folder),
            "manifest": str(manifest_path),
        }

        with open(reports_cam_folder / "extraction_summary.json", "w") as f:
            json.dump(cam_summary, f, indent=4)

        dataset_manifest.append(cam_summary)

        print(f"Saved frames: {saved}")
        print(f"Frames folder: {frames_folder}")

    dataset_manifest_path = reports_folder / "multicam_extraction_manifest.json"
    with open(dataset_manifest_path, "w") as f:
        json.dump(dataset_manifest, f, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Dataset manifest saved to: {dataset_manifest_path}")


if __name__ == "__main__":
    main()
