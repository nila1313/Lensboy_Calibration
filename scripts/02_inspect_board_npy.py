from pathlib import Path
import json
import numpy as np
import yaml


def describe_object(obj, name="object", indent=0):
    prefix = " " * indent

    print(f"{prefix}{name}:")
    print(f"{prefix}  Python type: {type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{prefix}  NumPy shape: {obj.shape}")
        print(f"{prefix}  NumPy dtype: {obj.dtype}")
        print(f"{prefix}  NumPy ndim: {obj.ndim}")

        if obj.size <= 50:
            print(f"{prefix}  Values:")
            print(obj)
        else:
            print(f"{prefix}  First values:")
            flat = obj.reshape(-1)
            print(flat[:20])

    elif isinstance(obj, dict):
        print(f"{prefix}  Dict keys:")
        for key in obj.keys():
            print(f"{prefix}    - {key}")

        print()
        for key, value in obj.items():
            describe_object(value, name=f"dict[{repr(key)}]", indent=indent + 4)

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}  Length: {len(obj)}")
        for i, value in enumerate(obj[:10]):
            describe_object(value, name=f"{name}[{i}]", indent=indent + 4)

    else:
        print(f"{prefix}  Repr:")
        print(f"{prefix}  {repr(obj)}")


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "preview": obj.reshape(-1)[:20].tolist() if obj.size > 0 else [],
        }

    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return {
        "type": str(type(obj)),
        "repr": repr(obj),
    }


def main():
    with open("configs/paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_folder = Path(config["single_input_folder"])
    output_folder = Path(config["single_output_folder"])

    board_path = input_folder / "board.npy"
    reports_folder = output_folder / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BOARD NPY INSPECTION")
    print("=" * 80)
    print(f"Board file: {board_path}")
    print()

    if not board_path.exists():
        raise FileNotFoundError(f"board.npy not found: {board_path}")

    try:
        data = np.load(board_path, allow_pickle=False)
        print("Loaded with allow_pickle=False")
    except Exception as e:
        print("Could not load with allow_pickle=False.")
        print(f"Reason: {e}")
        print()
        print("Trying allow_pickle=True because board.npy may contain a Python object.")
        print("Only do this for files you trust.")
        data = np.load(board_path, allow_pickle=True)

    print()
    describe_object(data, name="loaded_data")

    # If it is a 0-dimensional object array, unpack it.
    unpacked = data
    if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
        unpacked = data.item()
        print()
        print("=" * 80)
        print("UNPACKED OBJECT ARRAY")
        print("=" * 80)
        describe_object(unpacked, name="unpacked_data")

    summary = make_json_serializable(unpacked)

    summary_path = reports_folder / "board_npy_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
