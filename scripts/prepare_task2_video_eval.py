from pathlib import Path
from PIL import Image
import shutil

# -----------------------------
# Input VisDrone video sequence
# -----------------------------
seq_name = "uav0000297_00000_v"

src_seq_dir = Path("data/VisDrone-VID/VisDrone2019-VID-test-dev/sequences") / seq_name
src_ann_file = Path("data/VisDrone-VID/VisDrone2019-VID-test-dev/annotations") / f"{seq_name}.txt"

# -----------------------------
# Output YOLO-style dataset
# -----------------------------
out_root = Path("data/VisDrone-VID-YOLO-task2")
out_img_dir = out_root / "images" / "val" / seq_name
out_lbl_dir = out_root / "labels" / "val" / seq_name

out_img_dir.mkdir(parents=True, exist_ok=True)
out_lbl_dir.mkdir(parents=True, exist_ok=True)

# VisDrone classes:
# 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
# 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor
valid_categories = set(range(1, 11))

print("Source sequence:", src_seq_dir)
print("Annotation file:", src_ann_file)

if not src_seq_dir.exists():
    raise FileNotFoundError(src_seq_dir)

if not src_ann_file.exists():
    raise FileNotFoundError(src_ann_file)

# -----------------------------
# Copy or link images into YOLO-style folder
# -----------------------------
frames = sorted(src_seq_dir.glob("*.jpg"))
print("Frames found:", len(frames))

for img_path in frames:
    dst = out_img_dir / img_path.name
    if not dst.exists():
        shutil.copy2(img_path, dst)

# -----------------------------
# Create empty label files for every frame
# -----------------------------
for img_path in frames:
    (out_lbl_dir / f"{img_path.stem}.txt").write_text("")

# -----------------------------
# Read annotations and write YOLO labels
# VisDrone VID format:
# frame_index, target_id, x, y, w, h, score, category, truncation, occlusion
# -----------------------------
labels_by_frame = {}

with open(src_ann_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 10:
            continue

        frame_id = int(parts[0])
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        score = int(float(parts[6]))
        category = int(float(parts[7]))

        # Skip ignored regions and invalid categories
        if score == 0:
            continue
        if category not in valid_categories:
            continue
        if w <= 0 or h <= 0:
            continue

        frame_name = f"{frame_id:07d}.jpg"
        img_path = src_seq_dir / frame_name

        if not img_path.exists():
            continue

        img_w, img_h = Image.open(img_path).size

        # Convert VisDrone box to YOLO normalized format
        x_center = x + w / 2
        y_center = y + h / 2

        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # VisDrone category 1-10 -> YOLO class 0-9
        class_id = category - 1

        yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

        labels_by_frame.setdefault(frame_name, []).append(yolo_line)

# -----------------------------
# Write labels
# -----------------------------
total_boxes = 0

for frame_name, lines in labels_by_frame.items():
    label_path = out_lbl_dir / frame_name.replace(".jpg", ".txt")
    label_path.write_text("\n".join(lines) + "\n")
    total_boxes += len(lines)

print("YOLO image folder:", out_img_dir)
print("YOLO label folder:", out_lbl_dir)
print("Total ground-truth boxes written:", total_boxes)

# -----------------------------
# Write YAML file for YOLO validation
# -----------------------------
yaml_text = f"""
path: data/VisDrone-VID-YOLO-task2
train: images/val
val: images/val

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""

Path("visdrone_vid_task2.yaml").write_text(yaml_text.strip() + "\n")

print("Created YAML: visdrone_vid_task2.yaml")
