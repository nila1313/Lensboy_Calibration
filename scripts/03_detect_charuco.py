from pathlib import Path
import csv
import json
import cv2
import yaml
import numpy as np


def create_charuco_board(board_config):
    board_width = int(board_config["board_width"])
    board_height = int(board_config["board_height"])
    square_size = float(board_config["square_size_real"])
    marker_size = float(board_config["marker_size_real"])
    dictionary_type = int(board_config["dictionary_type"])

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)

    # OpenCV API changed across versions, so support both styles.
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (board_width, board_height),
            square_size,
            marker_size,
            aruco_dict
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            board_width,
            board_height,
            square_size,
            marker_size,
            aruco_dict
        )

    return board, aruco_dict


def detect_charuco_in_image(image, board, aruco_dict):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Newer OpenCV versions have CharucoDetector.
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=parameters
        )

        if marker_ids is not None and len(marker_ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board
            )
        else:
            charuco_corners, charuco_ids = None, None

    if charuco_corners is None or charuco_ids is None:
        return None, None, marker_corners, marker_ids

    charuco_corners = np.asarray(charuco_corners, dtype=np.float32)
    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def main():
    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/charuco_board.yaml", "r") as f:
        board_config = yaml.safe_load(f)

    output_folder = Path(paths_config["single_output_folder"])

    frames_folder = output_folder / "frames"
    detections_folder = output_folder / "detections"
    reports_folder = output_folder / "reports"
    debug_folder = output_folder / "debug" / "detected_corners"
    rejected_folder = output_folder / "debug" / "rejected_frames"

    detections_folder.mkdir(parents=True, exist_ok=True)
    reports_folder.mkdir(parents=True, exist_ok=True)
    debug_folder.mkdir(parents=True, exist_ok=True)
    rejected_folder.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_folder.glob("*.png"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_folder}")

    board, aruco_dict = create_charuco_board(board_config)

    print("=" * 80)
    print("CHARUCO DETECTION")
    print("=" * 80)
    print(f"Frames folder: {frames_folder}")
    print(f"Number of frames: {len(frame_paths)}")
    print(f"Board width: {board_config['board_width']}")
    print(f"Board height: {board_config['board_height']}")
    print(f"Square size: {board_config['square_size_real']}")
    print(f"Marker size: {board_config['marker_size_real']}")
    print(f"Dictionary type: {board_config['dictionary_type']}")
    print()

    all_detections = []
    summary_rows = []

    min_corners_for_use = 8
    valid_count = 0
    rejected_count = 0

    for i, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read: {frame_path}")
            continue

        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_in_image(
            image,
            board,
            aruco_dict
        )

        num_charuco = 0 if charuco_ids is None else len(charuco_ids)
        num_markers = 0 if marker_ids is None else len(marker_ids)

        is_valid = num_charuco >= min_corners_for_use

        if is_valid:
            valid_count += 1
        else:
            rejected_count += 1

        detection_record = {
            "frame_name": frame_path.name,
            "frame_path": str(frame_path),
            "num_markers": int(num_markers),
            "num_charuco_corners": int(num_charuco),
            "is_valid": bool(is_valid),
            "charuco_ids": charuco_ids.tolist() if charuco_ids is not None else [],
            "charuco_corners": charuco_corners.reshape(-1, 2).tolist() if charuco_corners is not None else [],
        }

        all_detections.append(detection_record)

        summary_rows.append([
            frame_path.name,
            num_markers,
            num_charuco,
            is_valid
        ])

        # Save debug image for useful frames and some rejected frames.
        annotated = image.copy()

        if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                annotated,
                charuco_corners,
                charuco_ids,
                (0, 255, 0)
            )

        label = f"markers={num_markers}, charuco={num_charuco}, valid={is_valid}"
        cv2.putText(
            annotated,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        if is_valid:
            # Save only every 10th valid debug image to avoid too many files.
            if valid_count % 10 == 0:
                cv2.imwrite(str(debug_folder / frame_path.name), annotated)
        else:
            # Save all rejected frames for inspection.
            cv2.imwrite(str(rejected_folder / frame_path.name), annotated)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frame_paths)} frames...")

    detections_json_path = detections_folder / "charuco_detections.json"
    with open(detections_json_path, "w") as f:
        json.dump(all_detections, f, indent=4)

    summary_csv_path = reports_folder / "charuco_detection_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_name",
            "num_markers",
            "num_charuco_corners",
            "is_valid"
        ])
        writer.writerows(summary_rows)

    metrics = {
        "total_frames": len(frame_paths),
        "valid_frames": valid_count,
        "rejected_frames": rejected_count,
        "min_corners_for_use": min_corners_for_use,
        "detections_json": str(detections_json_path),
        "summary_csv": str(summary_csv_path),
        "debug_detected_corners_folder": str(debug_folder),
        "debug_rejected_frames_folder": str(rejected_folder),
    }

    metrics_path = reports_folder / "charuco_detection_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Total frames: {len(frame_paths)}")
    print(f"Valid frames: {valid_count}")
    print(f"Rejected frames: {rejected_count}")
    print(f"Detection JSON: {detections_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Debug valid detections: {debug_folder}")
    print(f"Debug rejected frames: {rejected_folder}")


if __name__ == "__main__":
    main()
