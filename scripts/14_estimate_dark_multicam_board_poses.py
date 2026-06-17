from pathlib import Path
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

    if bool(board_config.get("legacy_pattern", False)) and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    return board


def get_charuco_target_points(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)

    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float32)

    raise RuntimeError("Could not get ChArUco 3D target points.")


def rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def rotation_error_degrees(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def main():
    with open("configs/dark_multicam_paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/dark_multicam_charuco_board.yaml", "r") as f:
        board_config = yaml.safe_load(f)

    output_folder = Path(paths_config["output_folder"])
    reports_folder = output_folder / "reports"
    poses_root = output_folder / "board_poses"

    reports_folder.mkdir(parents=True, exist_ok=True)
    poses_root.mkdir(parents=True, exist_ok=True)

    board = create_charuco_board(board_config)
    target_points_all = get_charuco_target_points(board)

    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]

    min_corners_for_pose = 8

    dataset_summary = []
    valid_pose_indices_by_cam = {}

    print("=" * 80)
    print("DARK MULTICAMERA BOARD POSE ESTIMATION")
    print("=" * 80)
    print(f"Minimum corners for pose estimation: {min_corners_for_pose}")
    print()

    for cam in cameras:
        print("-" * 80)
        print(f"Camera: {cam}")

        cam_folder = output_folder / cam
        detections_path = cam_folder / "detections" / "charuco_detections.json"
        metrics_path = cam_folder / "reports" / "lensboy_calibration_metrics.json"
        poses_cam_folder = poses_root / cam
        poses_cam_folder.mkdir(parents=True, exist_ok=True)

        if not detections_path.exists():
            print(f"{cam}: missing detections, skipping.")
            continue

        if not metrics_path.exists():
            print(f"{cam}: missing calibration metrics, skipping.")
            continue

        with open(detections_path, "r") as f:
            detections = json.load(f)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        K = np.asarray(metrics["K"], dtype=np.float64)

        if "distortion_coeffs" in metrics:
            dist = np.asarray(metrics["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
        else:
            dist = np.zeros((5, 1), dtype=np.float64)

        pose_records = []
        failed = 0

        for det in detections:
            if not det.get("is_valid", False):
                continue

            ids = np.asarray(det["charuco_ids"], dtype=np.int32).reshape(-1)
            corners = np.asarray(det["charuco_corners"], dtype=np.float32).reshape(-1, 2)

            if len(ids) < min_corners_for_pose:
                continue

            if len(ids) != len(corners):
                failed += 1
                continue

            object_points = target_points_all[ids].astype(np.float32)
            image_points = corners.astype(np.float32)

            # Try a stable planar PnP first; fall back if needed.
            success = False
            rvec = None
            tvec = None

            for flag in [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]:
                try:
                    ok, rv, tv = cv2.solvePnP(
                        object_points,
                        image_points,
                        K,
                        dist,
                        flags=flag
                    )
                    if ok:
                        success = True
                        rvec = rv
                        tvec = tv
                        break
                except Exception:
                    pass

            if not success:
                failed += 1
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

            T_cam_board = rvec_tvec_to_matrix(rvec, tvec)

            pose_records.append({
                "camera": cam,
                "frame_name": det["frame_name"],
                "frame_index": int(det["frame_index"]),
                "num_corners": int(len(ids)),
                "rvec": rvec.reshape(-1).tolist(),
                "tvec": tvec.reshape(-1).tolist(),
                "T_cam_board": T_cam_board.tolist(),
                "rms_reprojection_px": rms,
                "mean_reprojection_px": mean_error,
                "max_reprojection_px": max_error,
            })

        pose_records.sort(key=lambda x: x["frame_index"])

        poses_path = poses_cam_folder / "board_poses.json"
        with open(poses_path, "w") as f:
            json.dump(pose_records, f, indent=4)

        frame_indices = {p["frame_index"] for p in pose_records}
        valid_pose_indices_by_cam[cam] = frame_indices

        rms_values = [p["rms_reprojection_px"] for p in pose_records]

        summary = {
            "camera": cam,
            "poses_estimated": len(pose_records),
            "failed_pose_estimates": failed,
            "mean_pose_rms_px": float(np.mean(rms_values)) if rms_values else None,
            "median_pose_rms_px": float(np.median(rms_values)) if rms_values else None,
            "max_pose_rms_px": float(np.max(rms_values)) if rms_values else None,
            "poses_path": str(poses_path),
        }

        dataset_summary.append(summary)

        summary_path = poses_cam_folder / "board_pose_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"{cam}: poses estimated = {len(pose_records)}")
        print(f"{cam}: failed = {failed}")
        if rms_values:
            print(f"{cam}: mean RMS = {np.mean(rms_values):.3f} px")
            print(f"{cam}: median RMS = {np.median(rms_values):.3f} px")
            print(f"{cam}: max RMS = {np.max(rms_values):.3f} px")
        print(f"{cam}: saved poses to {poses_path}")

    # Common pose frame analysis.
    common_all = None
    for cam, indices in valid_pose_indices_by_cam.items():
        if common_all is None:
            common_all = set(indices)
        else:
            common_all &= set(indices)

    if common_all is None:
        common_all = set()

    pairwise = {}
    cams_available = sorted(valid_pose_indices_by_cam.keys())

    for i, cam_a in enumerate(cams_available):
        for cam_b in cams_available[i + 1:]:
            common = valid_pose_indices_by_cam[cam_a] & valid_pose_indices_by_cam[cam_b]
            pairwise[f"{cam_a}-{cam_b}"] = {
                "common_pose_frames": len(common),
                "common_frame_indices": sorted(common)
            }

    common_summary = {
        "common_pose_frames_all_cameras": len(common_all),
        "common_frame_indices_all_cameras": sorted(common_all),
        "pairwise_common_pose_frames": pairwise,
    }

    dataset_summary_path = reports_folder / "dark_multicam_board_pose_summary.json"
    with open(dataset_summary_path, "w") as f:
        json.dump(dataset_summary, f, indent=4)

    common_summary_path = reports_folder / "dark_multicam_common_pose_frames.json"
    with open(common_summary_path, "w") as f:
        json.dump(common_summary, f, indent=4)

    print()
    print("=" * 80)
    print("DATASET BOARD POSE SUMMARY")
    print("=" * 80)

    for item in dataset_summary:
        print(
            f"{item['camera']}: poses={item['poses_estimated']}, "
            f"mean_rms={item['mean_pose_rms_px']}"
        )

    print()
    print(f"Common pose frames across ALL cameras: {len(common_all)}")

    print()
    print("Pairwise common pose frames:")
    for key, value in pairwise.items():
        print(f"  {key}: {value['common_pose_frames']}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Board pose summary saved to: {dataset_summary_path}")
    print(f"Common pose summary saved to: {common_summary_path}")


if __name__ == "__main__":
    main()
