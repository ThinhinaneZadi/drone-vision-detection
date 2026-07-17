from ultralytics import YOLO
from pathlib import Path
import cv2
import csv

# -----------------------------
# Settings
# -----------------------------
model_path = "models/best_yolo11s_visdrone.pt"
video_path = "results/videos/external_drone_video.mp4"
candidate_csv = "results/metrics/task3_external_first_frame_candidates.csv"
candidate_id = 20

out_video = "results/videos/task3_external_candidate20_awning_tricycle.mp4"
out_csv = "results/metrics/task3_external_candidate20_awning_tricycle_trajectory.csv"

Path("results/videos").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper: IoU between two boxes
# -----------------------------
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0

# -----------------------------
# Read selected first-frame target
# -----------------------------
target_box = None
target_class = None

with open(candidate_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["candidate_id"]) == candidate_id:
            target_class = row["class"]
            target_box = [
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"])
            ]
            break

if target_box is None:
    raise ValueError(f"Candidate {candidate_id} not found in {candidate_csv}")

print("Selected external Task 3 target:")
print("Candidate ID:", candidate_id)
print("Class:", target_class)
print("Initial box:", target_box)

# -----------------------------
# Prepare video writer
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

# -----------------------------
# Run YOLO + ByteTrack
# -----------------------------
model = YOLO(model_path)

results_stream = model.track(
    source=video_path,
    imgsz=960,
    conf=0.15,
    iou=0.5,
    tracker="configs/bytetrack_long_buffer.yaml",
    device=0,
    stream=True,
    persist=True,
    verbose=False
)

target_track_id = None
trajectory_rows = []

for frame_idx, result in enumerate(results_stream, start=1):
    frame = result.orig_img.copy()
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        # Match candidate 14 in first frame to ByteTrack ID
        if frame_idx == 1:
            best_iou = 0
            best_track_id = None

            for box, track_id, cls_id in zip(xyxy, ids, cls):
                name = model.names[int(cls_id)]

                # Prefer same class, but the target_class is read from your candidate CSV
                if name != target_class:
                    continue

                box_int = [int(v) for v in box]
                score = iou(target_box, box_int)

                if score > best_iou:
                    best_iou = score
                    best_track_id = int(track_id)

            target_track_id = best_track_id
            print("Matched candidate", candidate_id, "to ByteTrack ID:", target_track_id)
            print("Matching IoU:", round(best_iou, 3))

        found = False

        # Draw only selected target ID
        for box, track_id, cls_id, conf in zip(xyxy, ids, cls, confs):
            if target_track_id is not None and int(track_id) == int(target_track_id):
                x1, y1, x2, y2 = [int(v) for v in box]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                found = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                label = f"Task 3 Target ID {target_track_id}: {model.names[int(cls_id)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                trajectory_rows.append([
                    frame_idx, 1, target_track_id, model.names[int(cls_id)],
                    round(float(conf), 3), x1, y1, x2, y2, cx, cy
                ])

        if not found:
            trajectory_rows.append([
                frame_idx, 0, target_track_id, target_class,
                "", "", "", "", "", "", ""
            ])
    else:
        trajectory_rows.append([
            frame_idx, 0, target_track_id, target_class,
            "", "", "", "", "", "", ""
        ])

    cv2.putText(frame, f"External Task 3 Single-Object Tracking | Candidate {candidate_id}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    writer.write(frame)

writer.release()

# -----------------------------
# Save trajectory CSV
# -----------------------------
with open(out_csv, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow([
        "frame", "found", "track_id", "class", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y"
    ])
    writer_csv.writerows(trajectory_rows)

print("\nSaved external Task 3 video:", out_video)
print("Saved external Task 3 trajectory CSV:", out_csv)
print("Target ByteTrack ID:", target_track_id)
