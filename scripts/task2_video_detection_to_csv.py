from ultralytics import YOLO
from pathlib import Path
import cv2
import csv

model_path = "models/best_yolo11s_visdrone.pt"
video_path = "results/videos/task_inputs/uav0000297_00000_v.mp4"

out_video = "results/videos/task2/task2_video_detection_yolo11s.mp4"
out_csv = "results/metrics/task2_video_detections.csv"

Path("results/videos/task2").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

rows = []
frame_idx = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_idx += 1

    result = model.predict(
        source=frame,
        imgsz=960,
        conf=0.30,
        iou=0.5,
        device=0,
        verbose=False
    )[0]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        rows.append([
            frame_idx,
            class_name,
            round(conf, 3),
            x1, y1, x2, y2,
            cx, cy
        ])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, "Task 2 Video Object Detection | YOLO11s",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    writer.write(frame)

cap.release()
writer.release()

with open(out_csv, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow([
        "frame", "class", "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y"
    ])
    writer_csv.writerows(rows)

print("Saved Task 2 video:", out_video)
print("Saved Task 2 detections CSV:", out_csv)
print("Total detections saved:", len(rows))
