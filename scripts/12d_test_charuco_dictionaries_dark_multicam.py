from pathlib import Path
import cv2
import yaml
import numpy as np


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}


def create_board(dict_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    board_width = 6
    board_height = 8
    square_size = 3.0
    marker_size = 1.8

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


def make_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 101
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return params


def detect_best_charuco(image, board, aruco_dict):
    params = make_params()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    else:
        detector = None

    best_markers = 0
    best_charuco = 0
    best_preprocess = None

    for prep_name, gray in preprocess_variants(image):
        if detector is not None:
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray,
                aruco_dict,
                parameters=params
            )

        num_markers = 0 if marker_ids is None else len(marker_ids)
        num_charuco = 0

        if marker_ids is not None and len(marker_ids) > 0:
            try:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners,
                    marker_ids,
                    gray,
                    board
                )
                if charuco_ids is not None:
                    num_charuco = len(charuco_ids)
            except Exception:
                num_charuco = 0

        if num_charuco > best_charuco:
            best_charuco = num_charuco
            best_markers = num_markers
            best_preprocess = prep_name

    return best_markers, best_charuco, best_preprocess


def main():
    with open("configs/dark_multicam_paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    output_folder = Path(config["output_folder"])

    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]

    print("=" * 80)
    print("CHARUCO DICTIONARY DIAGNOSTIC")
    print("=" * 80)

    for cam in cameras:
        frames_folder = output_folder / cam / "frames"
        frames = sorted(frames_folder.glob("*.jpg"))

        if len(frames) == 0:
            continue

        # Test all 475 frames because this dataset is short.
        print()
        print("=" * 60)
        print(cam)

        for dict_name, dict_id in DICTIONARIES.items():
            board, aruco_dict = create_board(dict_id)

            valid4 = 0
            valid6 = 0
            valid8 = 0
            max_charuco = 0
            max_markers = 0
            best_frame = None
            total_charuco = 0

            for frame_path in frames:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                markers, charuco, prep = detect_best_charuco(img, board, aruco_dict)

                total_charuco += charuco

                if charuco > max_charuco:
                    max_charuco = charuco
                    max_markers = markers
                    best_frame = frame_path.name

                if charuco >= 4:
                    valid4 += 1
                if charuco >= 6:
                    valid6 += 1
                if charuco >= 8:
                    valid8 += 1

            print(
                f"{dict_name}: "
                f"total_charuco={total_charuco}, "
                f"max_charuco={max_charuco}, "
                f"max_markers={max_markers}, "
                f"best_frame={best_frame}, "
                f"frames>=4={valid4}, "
                f"frames>=6={valid6}, "
                f"frames>=8={valid8}"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
