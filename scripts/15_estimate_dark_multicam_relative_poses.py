from pathlib import Path
import json
import argparse
import numpy as np
import yaml
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_poses(path):
    with open(path, "r") as f:
        records = json.load(f)

    by_frame = {}
    for r in records:
        by_frame[int(r["frame_index"])] = r

    return by_frame


def inv_T(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_matrix_to_quaternion(R):
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)

    return R


def average_quaternions(quats):
    quats = [np.asarray(q, dtype=np.float64) for q in quats]

    if len(quats) == 1:
        return quats[0] / np.linalg.norm(quats[0])

    ref = quats[0]

    aligned = []
    for q in quats:
        if np.dot(ref, q) < 0:
            q = -q
        aligned.append(q)

    A = np.zeros((4, 4), dtype=np.float64)
    for q in aligned:
        A += np.outer(q, q)

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    q_avg = eigenvectors[:, np.argmax(eigenvalues)]
    q_avg = q_avg / np.linalg.norm(q_avg)

    if q_avg[0] < 0:
        q_avg = -q_avg

    return q_avg


def rotation_error_deg(R_a, R_b):
    R = R_a @ R_b.T
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def estimate_relative_transform(ref_records, cam_records, max_pose_rms):
    common_frames = sorted(set(ref_records.keys()) & set(cam_records.keys()))

    candidates = []

    for frame_idx in common_frames:
        ref_pose = ref_records[frame_idx]
        cam_pose = cam_records[frame_idx]

        if ref_pose["rms_reprojection_px"] > max_pose_rms:
            continue

        if cam_pose["rms_reprojection_px"] > max_pose_rms:
            continue

        T_ref_board = np.asarray(ref_pose["T_cam_board"], dtype=np.float64)
        T_cam_board = np.asarray(cam_pose["T_cam_board"], dtype=np.float64)

        # X_ref = T_ref_board * X_board
        # X_cam = T_cam_board * X_board
        # Therefore X_ref = T_ref_board * inv(T_cam_board) * X_cam
        # This is T_ref_cam.
        T_ref_cam = T_ref_board @ inv_T(T_cam_board)

        candidates.append({
            "frame_index": frame_idx,
            "T_ref_cam": T_ref_cam,
            "ref_rms": ref_pose["rms_reprojection_px"],
            "cam_rms": cam_pose["rms_reprojection_px"],
        })

    if len(candidates) == 0:
        return None

    rotations = [c["T_ref_cam"][:3, :3] for c in candidates]
    translations = [c["T_ref_cam"][:3, 3] for c in candidates]

    quats = [rotation_matrix_to_quaternion(R) for R in rotations]
    q_avg = average_quaternions(quats)
    R_avg = quaternion_to_rotation_matrix(q_avg)

    # Median translation is robust to outliers.
    t_avg = np.median(np.stack(translations, axis=0), axis=0)

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    rot_errors = []
    trans_errors = []

    for c in candidates:
        R_c = c["T_ref_cam"][:3, :3]
        t_c = c["T_ref_cam"][:3, 3]

        rot_errors.append(rotation_error_deg(R_c, R_avg))
        trans_errors.append(float(np.linalg.norm(t_c - t_avg)))

    result = {
        "num_common_frames_before_filtering": len(common_frames),
        "num_candidates_after_rms_filtering": len(candidates),
        "frame_indices_used": [int(c["frame_index"]) for c in candidates],
        "T_ref_cam": T_avg.tolist(),
        "rotation_quaternion_wxyz": q_avg.tolist(),
        "translation": t_avg.tolist(),
        "median_rotation_error_deg": float(np.median(rot_errors)),
        "mean_rotation_error_deg": float(np.mean(rot_errors)),
        "max_rotation_error_deg": float(np.max(rot_errors)),
        "median_translation_error": float(np.median(trans_errors)),
        "mean_translation_error": float(np.mean(trans_errors)),
        "max_translation_error": float(np.max(trans_errors)),
        "mean_ref_pose_rms_px": float(np.mean([c["ref_rms"] for c in candidates])),
        "mean_cam_pose_rms_px": float(np.mean([c["cam_rms"] for c in candidates])),
    }

    return result


def plot_camera_rig(results, reference_camera, output_path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    positions = {}
    positions[reference_camera] = np.zeros(3)

    for cam, result in results.items():
        if cam == reference_camera:
            continue

        if result["status"] != "success":
            continue

        T_ref_cam = np.asarray(result["T_ref_cam"], dtype=np.float64)

        # Camera origin in reference coordinates:
        # X_ref = T_ref_cam * X_cam, and camera origin is [0,0,0,1].
        pos = T_ref_cam[:3, 3]
        positions[cam] = pos

    for cam, pos in positions.items():
        ax.scatter(pos[0], pos[1], pos[2], s=80)
        ax.text(pos[0], pos[1], pos[2], cam, fontsize=11)

    # Draw lines from reference to all cameras.
    ref_pos = positions[reference_camera]
    for cam, pos in positions.items():
        if cam == reference_camera:
            continue
        ax.plot(
            [ref_pos[0], pos[0]],
            [ref_pos[1], pos[1]],
            [ref_pos[2], pos[2]],
            linewidth=1
        )

    ax.set_title(f"Estimated camera rig in {reference_camera} coordinates")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Equal-ish axes.
    all_pos = np.stack(list(positions.values()), axis=0)
    center = all_pos.mean(axis=0)
    max_range = np.max(np.ptp(all_pos, axis=0))
    if max_range <= 0:
        max_range = 1.0

    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-camera", default="cam5")
    parser.add_argument("--max-pose-rms", type=float, default=5.0)
    parser.add_argument("--max-rotation-outlier-deg", type=float, default=5.0)
    args = parser.parse_args()

    with open("configs/dark_multicam_paths.yaml", "r") as f:
        paths_config = yaml.safe_load(f)

    output_folder = Path(paths_config["output_folder"])

    poses_root = output_folder / "board_poses"
    reports_folder = output_folder / "reports"
    plots_folder = output_folder / "plots"
    final_folder = output_folder / "final"

    reports_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)
    final_folder.mkdir(parents=True, exist_ok=True)

    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]
    ref_cam = args.reference_camera

    all_poses = {}

    for cam in cameras:
        pose_path = poses_root / cam / "board_poses.json"
        if not pose_path.exists():
            print(f"Missing poses for {cam}: {pose_path}")
            continue
        all_poses[cam] = load_poses(pose_path)

    if ref_cam not in all_poses:
        raise RuntimeError(f"Reference camera {ref_cam} has no board poses.")

    ref_records = all_poses[ref_cam]

    results = {}

    print("=" * 80)
    print("DARK MULTICAMERA RELATIVE POSE ESTIMATION")
    print("=" * 80)
    print(f"Reference camera: {ref_cam}")
    print(f"Max pose RMS filter: {args.max_pose_rms} px")
    print(f"Max rotation outlier filter: {args.max_rotation_outlier_deg} deg")
    print()

    results[ref_cam] = {
        "camera": ref_cam,
        "reference_camera": ref_cam,
        "status": "reference",
        "T_ref_cam": np.eye(4).tolist(),
        "num_candidates_after_rms_filtering": len(ref_records),
    }

    for cam in cameras:
        if cam == ref_cam:
            continue

        if cam not in all_poses:
            results[cam] = {
                "camera": cam,
                "reference_camera": ref_cam,
                "status": "missing_poses",
            }
            continue

        result = estimate_relative_transform(
            ref_records,
            all_poses[cam],
            max_pose_rms=args.max_pose_rms
        )

        if result is None:
            results[cam] = {
                "camera": cam,
                "reference_camera": ref_cam,
                "status": "no_common_candidates_after_filtering",
            }

            print(f"{ref_cam}-{cam}: no valid candidates after filtering")
            continue

        result["camera"] = cam
        result["reference_camera"] = ref_cam
        result["status"] = "success"

        results[cam] = result

        print("-" * 80)
        print(f"{ref_cam} <- {cam}")
        print(f"Common frames before filtering: {result['num_common_frames_before_filtering']}")
        print(f"Candidates after RMS filtering: {result['num_candidates_after_rms_filtering']}")
        print(f"Median rotation consistency: {result['median_rotation_error_deg']:.4f} deg")
        print(f"Mean rotation consistency: {result['mean_rotation_error_deg']:.4f} deg")
        print(f"Max rotation consistency: {result['max_rotation_error_deg']:.4f} deg")
        print(f"Median translation consistency: {result['median_translation_error']:.4f}")
        print(f"Mean ref pose RMS: {result['mean_ref_pose_rms_px']:.3f} px")
        print(f"Mean cam pose RMS: {result['mean_cam_pose_rms_px']:.3f} px")

    output_path = reports_folder / f"relative_camera_poses_reference_{ref_cam}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    final_path = final_folder / f"final_relative_camera_poses_reference_{ref_cam}.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=4)

    plot_path = plots_folder / f"camera_rig_reference_{ref_cam}.png"
    plot_camera_rig(results, ref_cam, plot_path)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Relative poses saved to: {output_path}")
    print(f"Final copy saved to: {final_path}")
    print(f"Rig plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
