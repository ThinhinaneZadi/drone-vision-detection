from ultralytics import YOLO
from pathlib import Path
import cv2
import csv

model_path = "models/best_yolo11s_visdrone.pt"
video_path = "results/videos/external_drone_video.mp4"

out_img = Path("results/images/task3_external/external_first_frame_candidates.jpg")
out_csv = Path("results/metrics/task3_external_first_frame_candidates.csv")

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError(f"Could not read first frame from {video_path}")

results = model.predict(
    source=frame,
    imgsz=640,
    conf=0.30,
    iou=0.5,
    device=0,
    verbose=False
)[0]

rows = []

for idx, box in enumerate(results.boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    name = model.names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    rows.append([idx, name, round(conf, 3), x1, y1, x2, y2])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{idx}: {name} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite(str(out_img), frame)

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "class", "confidence", "x1", "y1", "x2", "y2"])
    writer.writerows(rows)

print("Saved candidate image:", out_img)
print("Saved candidate CSV:", out_csv)
print("Number of detected candidates:", len(rows))

print("\nCandidates:")
for row in rows:
    print(row)
