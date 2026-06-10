from pathlib import Path
import cv2
import yaml
import numpy as np


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def preprocess_variants(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variants = [("gray", gray)]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))

    variants.append(("equalized", cv2.equalizeHist(gray)))

    gamma = 0.5
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    variants.append(("gamma_0.5", cv2.LUT(gray, table)))

    return variants


def detect_markers(gray, aruco_dict):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 101
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    return 0 if ids is None else len(ids)


def main():
    with open("configs/dark_multicam_paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    output_folder = Path(config["output_folder"])

    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]

    print("=" * 80)
    print("ARUCO DICTIONARY DIAGNOSTIC")
    print("=" * 80)

    for cam in cameras:
        frames_folder = output_folder / cam / "frames"
        frames = sorted(frames_folder.glob("*.jpg"))

        if len(frames) == 0:
            continue

        # Test 20 frames spread across the video.
        indices = np.linspace(0, len(frames) - 1, min(20, len(frames))).round().astype(int)
        selected = [frames[i] for i in indices]

        scores = {name: 0 for name in DICTIONARIES.keys()}

        for frame_path in selected:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            variants = preprocess_variants(img)

            for dict_name, dict_id in DICTIONARIES.items():
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

                best_for_frame = 0
                for prep_name, gray in variants:
                    n = detect_markers(gray, aruco_dict)
                    best_for_frame = max(best_for_frame, n)

                scores[dict_name] += best_for_frame

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print()
        print("=" * 60)
        print(cam)
        for name, score in sorted_scores[:8]:
            print(f"{name}: total markers across sampled frames = {score}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
