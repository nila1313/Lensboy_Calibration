from pathlib import Path
import json
import cv2
import yaml
import numpy as np
import lensboy as lb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    Frame = lb.Frame
except AttributeError:
    from lensboy.calibration.type_defs import Frame


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
        target_points = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        target_points = board.chessboardCorners
    else:
        raise RuntimeError("Could not get ChArUco board 3D points.")

    return np.asarray(target_points, dtype=np.float64)


def save_plot(plot_function, output_path, title):
    try:
        plt.close("all")
        plt.ioff()

        plot_function()

        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        axes = fig.get_axes()
        for ax in axes:
            ax.tick_params(axis="both", labelsize=9)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)

            legend = ax.get_legend()
            if legend is not None:
                for txt in legend.get_texts():
                    txt.set_fontsize(9)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: {output_path}")

    except Exception as e:
        plt.close("all")
        print(f"Could not save plot {output_path.name}: {e}")


def load_frames_from_detections(detections_path, target_points, min_corners):
    with open(detections_path, "r") as f:
        detections = json.load(f)

    frames = []
    used_frame_names = []

    for det in detections:
        if not det.get("is_valid", False):
            continue

        ids = np.asarray(det["charuco_ids"], dtype=np.int64).reshape(-1)
        corners = np.asarray(det["charuco_corners"], dtype=np.float64).reshape(-1, 2)

        if len(ids) < min_corners:
            continue

        if len(ids) != len(corners):
            continue

        if ids.min() < 0 or ids.max() >= len(target_points):
            continue

        frames.append(Frame(ids, corners))
        used_frame_names.append(det["frame_name"])

    return frames, used_frame_names


def get_image_size(frames_folder):
    frame_paths = sorted(frames_folder.glob("*.jpg"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_folder}")

    img = cv2.imread(str(frame_paths[0]))
    if img is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")

    h, w = img.shape[:2]
    return w, h


def main():
    with open("configs/dark_multicam_paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/dark_multicam_charuco_board.yaml", "r") as f:
        board_config = yaml.safe_load(f)

    output_folder = Path(paths_config["output_folder"])
    reports_folder = output_folder / "reports"
    models_root = output_folder / "models"

    reports_folder.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    board = create_charuco_board(board_config)
    target_points = get_charuco_target_points(board)

    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]

    min_corners_for_calibration = 8

    dataset_summary = []

    print("=" * 80)
    print("DARK MULTICAMERA PER-CAMERA INTRINSIC CALIBRATION")
    print("=" * 80)
    print(f"Minimum corners for calibration: {min_corners_for_calibration}")
    print(f"Target points shape: {target_points.shape}")
    print()

    for cam in cameras:
        print("-" * 80)
        print(f"Calibrating {cam}")

        cam_folder = output_folder / cam
        frames_folder = cam_folder / "frames"
        detections_path = cam_folder / "detections" / "charuco_detections.json"
        models_folder = cam_folder / "models"
        reports_cam_folder = cam_folder / "reports"
        plots_folder = cam_folder / "plots"

        models_folder.mkdir(parents=True, exist_ok=True)
        reports_cam_folder.mkdir(parents=True, exist_ok=True)
        plots_folder.mkdir(parents=True, exist_ok=True)

        image_width, image_height = get_image_size(frames_folder)

        frames, used_frame_names = load_frames_from_detections(
            detections_path,
            target_points,
            min_corners_for_calibration
        )

        print(f"{cam}: image size = {image_width} x {image_height}")
        print(f"{cam}: usable frames = {len(frames)}")

        if len(frames) < 10:
            print(f"{cam}: too few frames, skipping.")
            dataset_summary.append({
                "camera": cam,
                "status": "skipped_too_few_frames",
                "usable_frames": len(frames),
                "image_width": image_width,
                "image_height": image_height,
            })
            continue

        with open(reports_cam_folder / "lensboy_used_frames.txt", "w") as f:
            for name in used_frame_names:
                f.write(name + "\n")

        try:
            result = lb.calibrate_camera(
                target_points,
                frames,
                camera_model_config=lb.OpenCVConfig(
                    image_height=image_height,
                    image_width=image_width,
                ),
            )
        except Exception as e:
            print(f"{cam}: calibration failed: {e}")
            dataset_summary.append({
                "camera": cam,
                "status": "calibration_failed",
                "error": str(e),
                "usable_frames": len(frames),
                "image_width": image_width,
                "image_height": image_height,
            })
            continue

        model_path = models_folder / "lensboy_opencv_model.json"
        result.camera_model.save(str(model_path))

        metrics = {
            "camera": cam,
            "status": "success",
            "usable_frames": len(frames),
            "image_width": image_width,
            "image_height": image_height,
            "model_path": str(model_path),
            "board_config": board_config,
        }

        for attr in ["fx", "fy", "cx", "cy"]:
            if hasattr(result.camera_model, attr):
                try:
                    metrics[attr] = float(getattr(result.camera_model, attr))
                except Exception:
                    pass

        if hasattr(result.camera_model, "K"):
            try:
                metrics["K"] = result.camera_model.K().tolist()
            except Exception:
                pass

        if hasattr(result.camera_model, "distortion_coeffs"):
            try:
                metrics["distortion_coeffs"] = np.asarray(
                    result.camera_model.distortion_coeffs
                ).reshape(-1).tolist()
            except Exception:
                pass

        metrics_path = reports_cam_folder / "lensboy_calibration_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Also copy compact model/metrics to root models folder.
        root_model_path = models_root / f"{cam}_lensboy_opencv_model.json"
        result.camera_model.save(str(root_model_path))

        dataset_summary.append(metrics)

        save_plot(
            result.plot_residuals,
            plots_folder / "lensboy_residuals.png",
            f"{cam} Lensboy Residuals"
        )

        save_plot(
            result.plot_detection_coverage,
            plots_folder / "lensboy_detection_coverage.png",
            f"{cam} Detection Coverage"
        )

        save_plot(
            result.plot_inlier_coverage,
            plots_folder / "lensboy_inlier_coverage.png",
            f"{cam} Inlier Coverage"
        )

        save_plot(
            result.plot_per_image_rms,
            plots_folder / "lensboy_per_image_rms.png",
            f"{cam} Per-Image RMS"
        )

        print(f"{cam}: saved model to {model_path}")
        print(f"{cam}: saved metrics to {metrics_path}")

    dataset_summary_path = reports_folder / "dark_multicam_intrinsic_calibration_summary.json"
    with open(dataset_summary_path, "w") as f:
        json.dump(dataset_summary, f, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Dataset summary saved to: {dataset_summary_path}")

    print()
    print("Summary:")
    for item in dataset_summary:
        print(
            f"{item['camera']}: {item['status']}, "
            f"usable_frames={item.get('usable_frames')}"
        )


if __name__ == "__main__":
    main()
