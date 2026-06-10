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


def make_detector(aruco_dict):
    # OpenCV API changed between versions.
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    # Slightly permissive settings for dark / difficult frames.
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    else:
        detector = None

    return detector, parameters


def preprocess_variants(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variants = [("gray", gray)]

    # CLAHE often helps for dark frames.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))

    # Histogram equalization can sometimes help.
    variants.append(("equalized", cv2.equalizeHist(gray)))

    # Gamma/lightening variant for dark images.
    gamma = 0.65
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    bright = cv2.LUT(gray, table)
    variants.append(("gamma_0.65", bright))

    return variants


def detect_one_variant(gray, board, aruco_dict, detector, parameters):
    if detector is not None:
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=parameters
        )

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    try:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )
    except Exception:
        return None, None, marker_corners, marker_ids

    if charuco_corners is None or charuco_ids is None:
        return None, None, marker_corners, marker_ids

    charuco_corners = np.asarray(charuco_corners, dtype=np.float32)
    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def detect_charuco_best(image, board, aruco_dict, detector, parameters):
    best = {
        "num_charuco": 0,
        "preprocess": None,
        "charuco_corners": None,
        "charuco_ids": None,
        "marker_corners": None,
        "marker_ids": None,
    }

    for name, gray in preprocess_variants(image):
        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_one_variant(
            gray,
            board,
            aruco_dict,
            detector,
            parameters
        )

        num_charuco = 0 if charuco_ids is None else len(charuco_ids)

        if num_charuco > best["num_charuco"]:
            best = {
                "num_charuco": num_charuco,
                "preprocess": name,
                "charuco_corners": charuco_corners,
                "charuco_ids": charuco_ids,
                "marker_corners": marker_corners,
                "marker_ids": marker_ids,
            }

    return best


def frame_index_from_name(frame_name):
    # frame_000123.jpg -> 123
    stem = Path(frame_name).stem
    return int(stem.split("_")[-1])


def main():
    with open("configs/dark_multicam_paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/dark_multicam_charuco_board.yaml", "r") as f:
        board_config = yaml.safe_load(f)

    output_folder = Path(paths_config["output_folder"])
    reports_folder = output_folder / "reports"
    detections_root = output_folder / "detections"

    reports_folder.mkdir(parents=True, exist_ok=True)
    detections_root.mkdir(parents=True, exist_ok=True)

    board, aruco_dict = create_charuco_board(board_config)
    detector, parameters = make_detector(aruco_dict)

    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]

    min_corners_for_use = 8

    dataset_summary = []
    valid_frame_indices_by_cam = {}

    print("=" * 80)
    print("DARK MULTICAMERA CHARUCO DETECTION")
    print("=" * 80)
    print(f"Output folder: {output_folder}")
    print(f"Board: {board_config['board_width']} x {board_config['board_height']}")
    print(f"Square size: {board_config['square_size_real']}")
    print(f"Marker size: {board_config['marker_size_real']}")
    print(f"Dictionary type: {board_config['dictionary_type']}")
    print(f"Minimum ChArUco corners for valid frame: {min_corners_for_use}")
    print()

    for cam in cameras:
        cam_folder = output_folder / cam
        frames_folder = cam_folder / "frames"
        detections_folder = cam_folder / "detections"
        reports_cam_folder = cam_folder / "reports"
        debug_detected_folder = cam_folder / "debug" / "detected_corners"
        debug_rejected_folder = cam_folder / "debug" / "rejected_frames"

        detections_folder.mkdir(parents=True, exist_ok=True)
        reports_cam_folder.mkdir(parents=True, exist_ok=True)
        debug_detected_folder.mkdir(parents=True, exist_ok=True)
        debug_rejected_folder.mkdir(parents=True, exist_ok=True)

        # Clear old debug images.
        for p in debug_detected_folder.glob("*.jpg"):
            p.unlink()
        for p in debug_rejected_folder.glob("*.jpg"):
            p.unlink()

        frame_paths = sorted(frames_folder.glob("*.jpg"))

        if len(frame_paths) == 0:
            print(f"{cam}: no frames found, skipping.")
            continue

        print("-" * 80)
        print(f"Camera: {cam}")
        print(f"Frames: {len(frame_paths)}")

        all_detections = []
        summary_rows = []

        valid_count = 0
        rejected_count = 0
        valid_indices = []

        preprocess_counts = {}

        for i, frame_path in enumerate(frame_paths):
            image = cv2.imread(str(frame_path))

            if image is None:
                continue

            best = detect_charuco_best(
                image,
                board,
                aruco_dict,
                detector,
                parameters
            )

            charuco_ids = best["charuco_ids"]
            charuco_corners = best["charuco_corners"]
            marker_ids = best["marker_ids"]
            marker_corners = best["marker_corners"]
            preprocess_name = best["preprocess"]

            num_charuco = 0 if charuco_ids is None else len(charuco_ids)
            num_markers = 0 if marker_ids is None else len(marker_ids)

            is_valid = num_charuco >= min_corners_for_use
            frame_index = frame_index_from_name(frame_path.name)

            if is_valid:
                valid_count += 1
                valid_indices.append(frame_index)
            else:
                rejected_count += 1

            if preprocess_name is not None:
                preprocess_counts[preprocess_name] = preprocess_counts.get(preprocess_name, 0) + 1

            detection_record = {
                "camera": cam,
                "frame_name": frame_path.name,
                "frame_index": frame_index,
                "frame_path": str(frame_path),
                "num_markers": int(num_markers),
                "num_charuco_corners": int(num_charuco),
                "is_valid": bool(is_valid),
                "best_preprocess": preprocess_name,
                "charuco_ids": charuco_ids.tolist() if charuco_ids is not None else [],
                "charuco_corners": charuco_corners.reshape(-1, 2).tolist() if charuco_corners is not None else [],
            }

            all_detections.append(detection_record)

            summary_rows.append([
                cam,
                frame_path.name,
                frame_index,
                num_markers,
                num_charuco,
                is_valid,
                preprocess_name
            ])

            # Save debug images.
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

            label = f"{cam} markers={num_markers}, charuco={num_charuco}, valid={is_valid}, prep={preprocess_name}"
            cv2.putText(
                annotated,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            # Save every 20th valid frame only.
            if is_valid and valid_count % 20 == 0:
                cv2.imwrite(str(debug_detected_folder / frame_path.name), annotated)

            # Save first 20 rejected frames only, to avoid huge output.
            if not is_valid and rejected_count <= 20:
                cv2.imwrite(str(debug_rejected_folder / frame_path.name), annotated)

            if (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{len(frame_paths)} frames...")

        detections_path = detections_folder / "charuco_detections.json"
        with open(detections_path, "w") as f:
            json.dump(all_detections, f, indent=4)

        summary_csv_path = reports_cam_folder / "charuco_detection_summary.csv"
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "camera",
                "frame_name",
                "frame_index",
                "num_markers",
                "num_charuco_corners",
                "is_valid",
                "best_preprocess"
            ])
            writer.writerows(summary_rows)

        metrics = {
            "camera": cam,
            "total_frames": len(frame_paths),
            "valid_frames": valid_count,
            "rejected_frames": rejected_count,
            "min_corners_for_use": min_corners_for_use,
            "preprocess_counts": preprocess_counts,
            "detections_json": str(detections_path),
            "summary_csv": str(summary_csv_path),
        }

        metrics_path = reports_cam_folder / "charuco_detection_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        valid_frame_indices_by_cam[cam] = set(valid_indices)

        dataset_summary.append(metrics)

        print(f"{cam}: valid={valid_count}, rejected={rejected_count}")
        print(f"{cam}: detections saved to {detections_path}")

    # Save dataset-level detection summary.
    dataset_summary_path = reports_folder / "dark_multicam_charuco_detection_summary.json"
    with open(dataset_summary_path, "w") as f:
        json.dump(dataset_summary, f, indent=4)

    # Analyze common valid frame indices.
    common_all = None
    for cam, indices in valid_frame_indices_by_cam.items():
        if common_all is None:
            common_all = set(indices)
        else:
            common_all &= set(indices)

    if common_all is None:
        common_all = set()

    common_all_sorted = sorted(common_all)

    # Pairwise common counts.
    pairwise = {}
    cams_available = sorted(valid_frame_indices_by_cam.keys())

    for i, cam_a in enumerate(cams_available):
        for cam_b in cams_available[i + 1:]:
            common = valid_frame_indices_by_cam[cam_a] & valid_frame_indices_by_cam[cam_b]
            pairwise[f"{cam_a}-{cam_b}"] = {
                "common_valid_frames": len(common),
                "common_frame_indices": sorted(common)
            }

    common_summary = {
        "common_valid_frames_all_cameras": len(common_all_sorted),
        "common_frame_indices_all_cameras": common_all_sorted,
        "pairwise_common_valid_frames": pairwise,
    }

    common_summary_path = reports_folder / "dark_multicam_common_valid_frames.json"
    with open(common_summary_path, "w") as f:
        json.dump(common_summary, f, indent=4)

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    for item in dataset_summary:
        print(
            f"{item['camera']}: valid={item['valid_frames']} / {item['total_frames']}"
        )

    print()
    print(f"Common valid frames across ALL cameras: {len(common_all_sorted)}")
    if len(common_all_sorted) > 0:
        print(f"First common frame indices: {common_all_sorted[:20]}")

    print()
    print("Pairwise common valid frame counts:")
    for key, value in pairwise.items():
        print(f"  {key}: {value['common_valid_frames']}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Dataset detection summary saved to: {dataset_summary_path}")
    print(f"Common valid frames summary saved to: {common_summary_path}")


if __name__ == "__main__":
    main()
