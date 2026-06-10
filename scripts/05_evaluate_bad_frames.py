from pathlib import Path
import json
import csv
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

    return board


def get_charuco_target_points(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)

    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float32)

    raise RuntimeError("Could not get ChArUco 3D target points.")


def main():
    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/charuco_board.yaml", "r") as f:
        board_config = yaml.safe_load(f)

    output_folder = Path(paths_config["single_output_folder"])

    detections_path = output_folder / "detections" / "charuco_detections.json"
    metrics_path = output_folder / "reports" / "lensboy_calibration_metrics.json"

    frames_folder = output_folder / "frames"
    reports_folder = output_folder / "reports"
    debug_worst_folder = output_folder / "debug" / "worst_frames"

    reports_folder.mkdir(parents=True, exist_ok=True)
    debug_worst_folder.mkdir(parents=True, exist_ok=True)

    with open(detections_path, "r") as f:
        detections = json.load(f)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if "K" not in metrics:
        raise RuntimeError("K matrix not found in lensboy_calibration_metrics.json")

    K = np.asarray(metrics["K"], dtype=np.float64)

    if "distortion_coeffs" in metrics:
        dist = np.asarray(metrics["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
    else:
        print("Warning: distortion coefficients not found. Using zeros.")
        dist = np.zeros((5, 1), dtype=np.float64)

    board = create_charuco_board(board_config)
    target_points_all = get_charuco_target_points(board)

    rows = []

    for det in detections:
        if not det.get("is_valid", False):
            continue

        frame_name = det["frame_name"]
        ids = np.asarray(det["charuco_ids"], dtype=np.int32).reshape(-1)
        corners = np.asarray(det["charuco_corners"], dtype=np.float32).reshape(-1, 2)

        if len(ids) < 8:
            continue

        object_points = target_points_all[ids].astype(np.float32)
        image_points = corners.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            rows.append([frame_name, len(ids), None, None, None, "solvePnP_failed"])
            continue

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            dist
        )

        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(projected - image_points, axis=1)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))

        rows.append([
            frame_name,
            len(ids),
            rms,
            mean_error,
            max_error,
            "ok"
        ])

    rows_ok = [r for r in rows if r[2] is not None]
    rows_ok.sort(key=lambda r: r[2], reverse=True)

    csv_path = reports_folder / "per_frame_reprojection_errors.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_name",
            "num_corners",
            "rms_error_px",
            "mean_error_px",
            "max_error_px",
            "status"
        ])
        writer.writerows(rows_ok)

    worst_txt = reports_folder / "worst_frames_by_reprojection_error.txt"
    with open(worst_txt, "w") as f:
        f.write("Worst frames by reprojection RMS error\n")
        f.write("=" * 60 + "\n\n")
        for r in rows_ok[:25]:
            f.write(
                f"{r[0]} | corners={r[1]} | RMS={r[2]:.3f}px | "
                f"mean={r[3]:.3f}px | max={r[4]:.3f}px\n"
            )

    # Clear old worst-frame images first.
    for old_file in debug_worst_folder.glob("*.png"):
        old_file.unlink()

    # Save visual copies of worst frames.
    for r in rows_ok[:25]:
        frame_name = r[0]
        frame_path = frames_folder / frame_name
        img = cv2.imread(str(frame_path))

        if img is None:
            continue

        text = f"RMS={r[2]:.2f}px  mean={r[3]:.2f}px  max={r[4]:.2f}px"
        cv2.putText(
            img,
            text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            4,
            cv2.LINE_AA
        )

        cv2.imwrite(str(debug_worst_folder / frame_name), img)

    rms_values = [r[2] for r in rows_ok]

    summary = {
        "evaluated_frames": len(rows_ok),
        "mean_rms_px": float(np.mean(rms_values)) if rms_values else None,
        "median_rms_px": float(np.median(rms_values)) if rms_values else None,
        "max_rms_px": float(np.max(rms_values)) if rms_values else None,
        "frames_above_2px": int(sum(v > 2 for v in rms_values)),
        "frames_above_5px": int(sum(v > 5 for v in rms_values)),
        "frames_above_10px": int(sum(v > 10 for v in rms_values)),
        "frames_above_20px": int(sum(v > 20 for v in rms_values)),
    }

    summary_path = reports_folder / "per_frame_error_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("PER-FRAME REPROJECTION EVALUATION")
    print("=" * 80)
    print(f"Evaluated frames: {summary['evaluated_frames']}")
    print(f"Mean RMS: {summary['mean_rms_px']:.3f} px")
    print(f"Median RMS: {summary['median_rms_px']:.3f} px")
    print(f"Max RMS: {summary['max_rms_px']:.3f} px")
    print(f"Frames above 2 px: {summary['frames_above_2px']}")
    print(f"Frames above 5 px: {summary['frames_above_5px']}")
    print(f"Frames above 10 px: {summary['frames_above_10px']}")
    print(f"Frames above 20 px: {summary['frames_above_20px']}")
    print()
    print("Worst 15 frames:")
    for r in rows_ok[:15]:
        print(
            f"{r[0]} | corners={r[1]} | RMS={r[2]:.3f}px | "
            f"mean={r[3]:.3f}px | max={r[4]:.3f}px"
        )
    print()
    print(f"CSV saved to: {csv_path}")
    print(f"Worst frame list saved to: {worst_txt}")
    print(f"Worst frame images saved to: {debug_worst_folder}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
