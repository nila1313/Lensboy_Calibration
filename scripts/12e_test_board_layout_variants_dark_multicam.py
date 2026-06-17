from pathlib import Path
import cv2
import yaml
import numpy as np


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}


BOARD_VARIANTS = [
    {"name": "6x8_marker1.8", "width": 6, "height": 8, "square": 3.0, "marker": 1.8},
    {"name": "8x6_marker1.8", "width": 8, "height": 6, "square": 3.0, "marker": 1.8},
    {"name": "6x8_marker0.6", "width": 6, "height": 8, "square": 3.0, "marker": 0.6},
    {"name": "8x6_marker0.6", "width": 8, "height": 6, "square": 3.0, "marker": 0.6},
    {"name": "6x8_unit_marker0.6", "width": 6, "height": 8, "square": 1.0, "marker": 0.6},
    {"name": "8x6_unit_marker0.6", "width": 8, "height": 6, "square": 1.0, "marker": 0.6},
]


def create_board(width, height, square_size, marker_size, aruco_dict, legacy=False):
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (width, height),
            square_size,
            marker_size,
            aruco_dict
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            width,
            height,
            square_size,
            marker_size,
            aruco_dict
        )

    # Newer OpenCV has a legacy pattern setting.
    if legacy and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    return board


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
    print("BOARD LAYOUT VARIANT DIAGNOSTIC")
    print("=" * 80)

    global_results = []

    for dict_name, dict_id in DICTIONARIES.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        for variant in BOARD_VARIANTS:
            for legacy in [False, True]:
                config_name = f"{dict_name}_{variant['name']}_legacy{legacy}"

                print()
                print("=" * 80)
                print(config_name)
                print("=" * 80)

                total_valid4_all = 0
                total_valid6_all = 0
                total_valid8_all = 0
                total_charuco_all = 0
                max_charuco_all = 0
                best_overall = None

                for cam in cameras:
                    frames_folder = output_folder / cam / "frames"
                    frames = sorted(frames_folder.glob("*.jpg"))

                    board = create_board(
                        variant["width"],
                        variant["height"],
                        variant["square"],
                        variant["marker"],
                        aruco_dict,
                        legacy=legacy
                    )

                    valid4 = 0
                    valid6 = 0
                    valid8 = 0
                    total_charuco = 0
                    max_charuco = 0
                    max_markers = 0
                    best_frame = None
                    best_prep = None

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
                            best_prep = prep

                        if charuco >= 4:
                            valid4 += 1
                        if charuco >= 6:
                            valid6 += 1
                        if charuco >= 8:
                            valid8 += 1

                    total_valid4_all += valid4
                    total_valid6_all += valid6
                    total_valid8_all += valid8
                    total_charuco_all += total_charuco

                    if max_charuco > max_charuco_all:
                        max_charuco_all = max_charuco
                        best_overall = f"{cam} {best_frame} prep={best_prep} markers={max_markers}"

                    print(
                        f"{cam}: "
                        f"total_charuco={total_charuco}, "
                        f"max_charuco={max_charuco}, "
                        f"best_frame={best_frame}, "
                        f"prep={best_prep}, "
                        f"frames>=4={valid4}, "
                        f"frames>=6={valid6}, "
                        f"frames>=8={valid8}"
                    )

                global_results.append({
                    "config": config_name,
                    "total_charuco_all": total_charuco_all,
                    "max_charuco_all": max_charuco_all,
                    "total_frames_ge4": total_valid4_all,
                    "total_frames_ge6": total_valid6_all,
                    "total_frames_ge8": total_valid8_all,
                    "best_overall": best_overall,
                })

                print()
                print(
                    f"SUMMARY {config_name}: "
                    f"total_charuco_all={total_charuco_all}, "
                    f"max_charuco_all={max_charuco_all}, "
                    f"frames>=4 total={total_valid4_all}, "
                    f"frames>=6 total={total_valid6_all}, "
                    f"frames>=8 total={total_valid8_all}, "
                    f"best={best_overall}"
                )

    print()
    print("=" * 80)
    print("TOP CONFIGURATIONS")
    print("=" * 80)

    global_results.sort(
        key=lambda x: (
            x["total_frames_ge8"],
            x["total_frames_ge6"],
            x["total_frames_ge4"],
            x["max_charuco_all"],
            x["total_charuco_all"]
        ),
        reverse=True
    )

    for item in global_results[:10]:
        print(
            f"{item['config']}: "
            f"frames>=8={item['total_frames_ge8']}, "
            f"frames>=6={item['total_frames_ge6']}, "
            f"frames>=4={item['total_frames_ge4']}, "
            f"max_charuco={item['max_charuco_all']}, "
            f"total_charuco={item['total_charuco_all']}, "
            f"best={item['best_overall']}"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
