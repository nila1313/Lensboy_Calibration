from pathlib import Path
import csv
import json
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Remove valid frames with RMS error above this threshold."
    )
    args = parser.parse_args()

    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    output_folder = Path(paths_config["single_output_folder"])

    detections_path = output_folder / "detections" / "charuco_detections.json"
    errors_csv_path = output_folder / "reports" / "per_frame_reprojection_errors.csv"

    clean_detections_path = output_folder / "detections" / f"charuco_detections_clean_rms{args.threshold:g}.json"
    excluded_txt_path = output_folder / "reports" / f"excluded_frames_rms_above_{args.threshold:g}.txt"

    with open(detections_path, "r") as f:
        detections = json.load(f)

    bad_frames = set()

    with open(errors_csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_name = row["frame_name"]
            rms = float(row["rms_error_px"])

            if rms > args.threshold:
                bad_frames.add(frame_name)

    clean_detections = []

    original_valid = 0
    clean_valid = 0

    for det in detections:
        if det.get("is_valid", False):
            original_valid += 1

        if det["frame_name"] in bad_frames:
            det = dict(det)
            det["is_valid"] = False
            det["rejected_by_cleaning"] = True
            det["cleaning_reason"] = f"RMS reprojection error above {args.threshold:g}px"
        else:
            if det.get("is_valid", False):
                clean_valid += 1

        clean_detections.append(det)

    with open(clean_detections_path, "w") as f:
        json.dump(clean_detections, f, indent=4)

    with open(excluded_txt_path, "w") as f:
        f.write(f"Excluded frames with RMS reprojection error > {args.threshold:g}px\n")
        f.write("=" * 70 + "\n\n")
        for frame_name in sorted(bad_frames):
            f.write(frame_name + "\n")

    print("=" * 80)
    print("CLEAN DETECTION FILE CREATED")
    print("=" * 80)
    print(f"Threshold: {args.threshold:g} px")
    print(f"Original valid frames: {original_valid}")
    print(f"Excluded frames: {len(bad_frames)}")
    print(f"Clean valid frames: {clean_valid}")
    print()
    print(f"Clean detections saved to:")
    print(clean_detections_path)
    print()
    print(f"Excluded frame list saved to:")
    print(excluded_txt_path)


if __name__ == "__main__":
    main()
