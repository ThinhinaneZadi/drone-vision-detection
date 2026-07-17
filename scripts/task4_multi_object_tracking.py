from ultralytics import YOLO
from pathlib import Path
import cv2
import csv

# -----------------------------
# Settings
# -----------------------------
model_path = "models/best_yolo11s_visdrone.pt"
video_path = "results/videos/task_inputs/uav0000297_00000_v.mp4"

out_video = "results/videos/task4/task4_multi_object_tracking_bytetrack.mp4"
out_csv = "results/metrics/task4_multi_object_trajectories.csv"

Path("results/videos/task4").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

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
    imgsz=640,
    conf=0.30,
    iou=0.5,
    tracker="bytetrack.yaml",
    device=0,
    stream=True,
    persist=True,
    verbose=False
)

trajectory_rows = []

for frame_idx, result in enumerate(results_stream, start=1):
    frame = result.orig_img.copy()
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(xyxy, ids, cls, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_name = model.names[int(cls_id)]

            # Save trajectory row
            trajectory_rows.append([
                frame_idx,
                int(track_id),
                class_name,
                round(float(conf), 3),
                x1, y1, x2, y2,
                cx, cy
            ])

            # Draw all tracked objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            label = f"ID {track_id}: {class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, "Task 4 Multi-Object Tracking | YOLO11s + ByteTrack",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    writer.write(frame)

writer.release()

# -----------------------------
# Save all object trajectories
# -----------------------------
with open(out_csv, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow([
        "frame", "track_id", "class", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y"
    ])
    writer_csv.writerows(trajectory_rows)

print("Saved Task 4 video:", out_video)
print("Saved Task 4 trajectory CSV:", out_csv)
print("Total tracked detections saved:", len(trajectory_rows))
