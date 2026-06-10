from pathlib import Path
import yaml

CONFIG_PATH = Path("configs/paths.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

input_folder = Path(config["single_input_folder"])

print("=" * 80)
print("INPUT FOLDER")
print("=" * 80)
print(input_folder)
print()

if not input_folder.exists():
    raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".MOV", ".MP4"]
image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

videos = []
images = []

for path in input_folder.rglob("*"):
    if path.suffix in video_extensions:
        videos.append(path)
    elif path.suffix in image_extensions:
        images.append(path)

print(f"Found videos: {len(videos)}")
for v in videos:
    print("VIDEO:", v)

print()
print(f"Found images: {len(images)}")
for img in images[:20]:
    print("IMAGE:", img)

if len(images) > 20:
    print(f"... and {len(images) - 20} more images")

print()
print("=" * 80)
print("DONE")
print("=" * 80)