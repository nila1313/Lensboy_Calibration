from pathlib import Path
import json
import cv2
import yaml
import numpy as np


def describe_object(obj, name="object", indent=0):
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  type: {type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{prefix}  shape: {obj.shape}")
        print(f"{prefix}  dtype: {obj.dtype}")
        print(f"{prefix}  ndim: {obj.ndim}")
        if obj.size <= 50:
            print(f"{prefix}  values: {obj}")
        else:
            print(f"{prefix}  first values: {obj.reshape(-1)[:20]}")

    elif isinstance(obj, dict):
        print(f"{prefix}  keys:")
        for key in obj.keys():
            print(f"{prefix}    - {key}")
        print()
        for key, value in obj.items():
            describe_object(value, f"dict[{repr(key)}]", indent + 4)

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}  length: {len(obj)}")
        for i, value in enumerate(obj[:10]):
            describe_object(value, f"{name}[{i}]", indent + 4)

    else:
        print(f"{prefix}  repr: {repr(obj)}")


def main():
    with open("configs/dark_multicam_paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_folder = Path(config["input_folder"])
    output_folder = Path(config["output_folder"])
    reports_folder = output_folder / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DARK MULTICAMERA DATASET INSPECTION")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print()

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    videos = sorted([
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ])

    print("=" * 80)
    print("VIDEOS")
    print("=" * 80)
    print(f"Found videos: {len(videos)}")

    video_info = []

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else None

        cap.release()

        info = {
            "filename": video_path.name,
            "path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }

        video_info.append(info)

        print()
        print(f"VIDEO: {video_path.name}")
        print(f"  frames: {total_frames}")
        print(f"  fps: {fps}")
        print(f"  size: {width} x {height}")
        print(f"  duration: {duration:.2f} sec" if duration else "  duration: unknown")

    with open(reports_folder / "dark_multicam_video_info.json", "w") as f:
        json.dump(video_info, f, indent=4)

    print()
    print("=" * 80)
    print("BOARD NPY")
    print("=" * 80)

    board_files = sorted(input_folder.glob("*.npy"))
    print(f"Found npy files: {len(board_files)}")

    for board_path in board_files:
        print()
        print(f"NPY file: {board_path}")

        try:
            data = np.load(board_path, allow_pickle=False)
            print("Loaded with allow_pickle=False")
        except Exception as e:
            print(f"Could not load with allow_pickle=False: {e}")
            print("Trying allow_pickle=True...")
            data = np.load(board_path, allow_pickle=True)

        describe_object(data, "loaded_data")

        if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
            unpacked = data.item()
            print()
            print("UNPACKED:")
            describe_object(unpacked, "unpacked_data")

            summary_path = reports_folder / f"{board_path.stem}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {str(k): v for k, v in unpacked.items()},
                    f,
                    indent=4
                )
            print(f"Board summary saved to: {summary_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Video info saved to: {reports_folder / 'dark_multicam_video_info.json'}")


if __name__ == "__main__":
    main()
