from pathlib import Path
import json
import cv2
import yaml
import numpy as np
import lensboy as lb
import matplotlib
matplotlib.use('Agg')
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

    return board


def get_charuco_target_points(board):
    if hasattr(board, "getChessboardCorners"):
        target_points = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        target_points = board.chessboardCorners
    else:
        raise RuntimeError("Could not get ChArUco board 3D corner points from OpenCV board.")

    target_points = np.asarray(target_points, dtype=np.float64)

    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise RuntimeError(f"Unexpected target_points shape: {target_points.shape}")

    return target_points


def save_plot(plot_function, output_path, title):
    try:
        plt.close("all")
        plt.ioff()

        plot_function()

        fig = plt.gcf()
        fig.set_size_inches(10, 7)

        axes = fig.get_axes()
        for ax in axes:
            ax.set_title(ax.get_title(), fontsize=14)
            ax.tick_params(axis="both", labelsize=11)
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)

            legend = ax.get_legend()
            if legend is not None:
                legend.set_bbox_to_anchor((0.5, 1.15))
                legend._loc = 9  # upper center
                for txt in legend.get_texts():
                    txt.set_fontsize(11)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved clearer plot: {output_path}")
    except Exception as e:
        plt.close("all")
        print(f"Could not save plot {output_path.name}: {e}")



def save_per_image_rms_plots(plot_function, output_path_full, output_path_zoom):
    try:
        # Full plot, but taller so y-axis image labels are readable.
        plt.close("all")
        plt.ioff()

        plot_function()

        fig = plt.gcf()
        fig.set_size_inches(11, 16)

        axes = fig.get_axes()
        for ax in axes:
            ax.set_title("Per-image residual RMS", fontsize=16)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=7)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)

            legend = ax.get_legend()
            if legend is not None:
                legend.set_bbox_to_anchor((0.5, 1.05))
                legend._loc = 9
                for txt in legend.get_texts():
                    txt.set_fontsize(12)

        plt.tight_layout()
        fig.savefig(output_path_full, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved full RMS plot: {output_path_full}")

        # Zoomed plot: same data, but x-axis limited so normal frames are visible.
        plt.close("all")
        plt.ioff()

        plot_function()

        fig = plt.gcf()
        fig.set_size_inches(11, 16)

        axes = fig.get_axes()
        for ax in axes:
            ax.set_xlim(0, 10)
            ax.set_title("Per-image residual RMS zoomed: 0–10 px", fontsize=16)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=7)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)

            legend = ax.get_legend()
            if legend is not None:
                legend.set_bbox_to_anchor((0.5, 1.05))
                legend._loc = 9
                for txt in legend.get_texts():
                    txt.set_fontsize(12)

        plt.tight_layout()
        fig.savefig(output_path_zoom, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved zoomed RMS plot: {output_path_zoom}")

    except Exception as e:
        plt.close("all")
        print(f"Could not save per-image RMS plots: {e}")


def main():
    with open("configs/paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/charuco_board.yaml", "r") as f:
        board_config = yaml.safe_load(f)

    output_folder = Path(paths_config["single_output_folder"])

    frames_folder = output_folder / "frames"
    detections_folder = output_folder / "detections"
    models_folder = output_folder / "models"
    plots_folder = output_folder / "plots"
    reports_folder = output_folder / "reports"

    models_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)
    reports_folder.mkdir(parents=True, exist_ok=True)

    clean_detection_path = detections_folder / "charuco_detections_clean_rms5.json"
    if clean_detection_path.exists():
        detection_path = clean_detection_path
        print("Using CLEAN detection file.")
    else:
        clean_detection_path = detections_folder / "charuco_detections_clean_rms5.json"
    if clean_detection_path.exists():
        detection_path = clean_detection_path
        print("Using CLEAN detection file:", detection_path)
    else:
        detection_path = detections_folder / "charuco_detections.json"
        print("Using ORIGINAL detection file:", detection_path)
        print("Using original detection file.")

    if not detection_path.exists():
        raise FileNotFoundError(f"Detection file not found: {detection_path}")

    print("=" * 80)
    print("LENSBOY SINGLE CAMERA CALIBRATION")
    print("=" * 80)
    print("Using precomputed valid ChArUco detections.")
    print(f"Detection file: {detection_path}")
    print()

    with open(detection_path, "r") as f:
        detections = json.load(f)

    # Get image size from first readable frame.
    frame_paths = sorted(frames_folder.glob("*.png"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_folder}")

    first_img = cv2.imread(str(frame_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    image_height, image_width = first_img.shape[:2]

    board = create_charuco_board(board_config)
    target_points = get_charuco_target_points(board)

    frames = []
    used_frame_names = []
    skipped_frame_names = []

    min_corners_for_calibration = 8

    for det in detections:
        if not det.get("is_valid", False):
            continue

        ids = np.asarray(det["charuco_ids"], dtype=np.int64)
        corners = np.asarray(det["charuco_corners"], dtype=np.float64)

        if ids.ndim != 1:
            ids = ids.reshape(-1)

        if corners.ndim != 2 or corners.shape[1] != 2:
            skipped_frame_names.append(det["frame_name"])
            continue

        if len(ids) != len(corners):
            skipped_frame_names.append(det["frame_name"])
            continue

        if len(ids) < min_corners_for_calibration:
            skipped_frame_names.append(det["frame_name"])
            continue

        # Safety check: ChArUco ids must index target_points.
        if ids.min() < 0 or ids.max() >= len(target_points):
            skipped_frame_names.append(det["frame_name"])
            continue

        frames.append(Frame(ids, corners))
        used_frame_names.append(det["frame_name"])

    print(f"Image width: {image_width}")
    print(f"Image height: {image_height}")
    print(f"Target 3D ChArUco points: {target_points.shape}")
    print(f"Usable valid frames for Lensboy: {len(frames)}")
    print(f"Skipped valid-looking frames: {len(skipped_frame_names)}")
    print()

    if len(frames) < 10:
        raise RuntimeError(f"Too few usable frames for calibration: {len(frames)}")

    with open(reports_folder / "lensboy_used_frames.txt", "w") as f:
        for name in used_frame_names:
            f.write(name + "\n")

    with open(reports_folder / "lensboy_skipped_frames.txt", "w") as f:
        for name in skipped_frame_names:
            f.write(name + "\n")

    print("Running Lensboy OpenCV-style calibration...")

    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=lb.OpenCVConfig(
            image_height=image_height,
            image_width=image_width,
        ),
    )

    model_path = models_folder / "lensboy_opencv_model.json"
    result.camera_model.save(str(model_path))

    print(f"Saved camera model: {model_path}")

    metrics = {
        "total_extracted_frames": len(frame_paths),
        "valid_detection_frames_from_previous_step": sum(1 for d in detections if d.get("is_valid", False)),
        "lensboy_usable_frames": len(frames),
        "skipped_valid_looking_frames": len(skipped_frame_names),
        "image_width": image_width,
        "image_height": image_height,
        "target_points_shape": list(target_points.shape),
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
            coeffs = result.camera_model.distortion_coeffs
            metrics["distortion_coeffs"] = np.asarray(coeffs).reshape(-1).tolist()
        except Exception:
            pass

    metrics_path = reports_folder / "lensboy_calibration_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics: {metrics_path}")

    print()
    print("Saving evaluation plots...")

    save_plot(
        result.plot_residuals,
        plots_folder / "lensboy_residuals.png",
        "Lensboy Residuals"
    )

    save_plot(
        result.plot_detection_coverage,
        plots_folder / "lensboy_detection_coverage.png",
        "Lensboy Detection Coverage"
    )

    save_plot(
        result.plot_inlier_coverage,
        plots_folder / "lensboy_inlier_coverage.png",
        "Lensboy Inlier Coverage"
    )

    save_plot(
        result.plot_distortion_grid,
        plots_folder / "lensboy_distortion_grid.png",
        "Lensboy Distortion Grid"
    )

    save_per_image_rms_plots(
        result.plot_per_image_rms,
        plots_folder / "lensboy_per_image_rms_full.png",
        plots_folder / "lensboy_per_image_rms_zoom_0_10px.png"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Lensboy usable frames: {len(frames)}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plots_folder}")


if __name__ == "__main__":
    main()
