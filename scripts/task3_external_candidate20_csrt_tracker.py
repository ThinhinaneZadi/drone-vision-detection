from pathlib import Path
import cv2
import csv

# -----------------------------
# Settings
# -----------------------------
video_path = "results/videos/external_drone_video.mp4"
candidate_csv = "results/metrics/task3_external_first_frame_candidates.csv"
candidate_id = 20

out_video = "results/videos/task3_external_candidate20_csrt_tracker.mp4"
out_csv = "results/metrics/task3_external_candidate20_csrt_trajectory.csv"

Path("results/videos").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load first-frame target box from YOLO candidate CSV
# -----------------------------
target_box = None
target_class = None

with open(candidate_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["candidate_id"]) == candidate_id:
            target_class = row["class"]
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            target_box = (x1, y1, x2 - x1, y2 - y1)  # OpenCV format: x, y, w, h
            break

if target_box is None:
    raise ValueError(f"Candidate {candidate_id} not found in {candidate_csv}")

print("Selected Task 3 target:")
print("Candidate ID:", candidate_id)
print("Class:", target_class)
print("Initial OpenCV box x,y,w,h:", target_box)

# -----------------------------
# Create OpenCV tracker
# -----------------------------
def create_tracker():
    # Try modern OpenCV first
    if hasattr(cv2, "TrackerCSRT_create"):
        print("Using cv2.TrackerCSRT_create()")
        return cv2.TrackerCSRT_create()

    # Try legacy OpenCV
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        print("Using cv2.legacy.TrackerCSRT_create()")
        return cv2.legacy.TrackerCSRT_create()

    # Fallbacks
    if hasattr(cv2, "TrackerKCF_create"):
        print("CSRT not available. Using KCF tracker.")
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        print("CSRT not available. Using legacy KCF tracker.")
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        "No CSRT/KCF tracker found. Install with: pip install opencv-contrib-python"
    )

tracker = create_tracker()

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
success, first_frame = cap.read()

if not success:
    raise RuntimeError(f"Could not read video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

# Initialize tracker with candidate 20 box
tracker.init(first_frame, target_box)

trajectory_rows = []

frame_idx = 1
while True:
    if frame_idx == 1:
        frame = first_frame.copy()
        ok = True
        box = target_box
    else:
        success, frame = cap.read()
        if not success:
            break

        ok, box = tracker.update(frame)

    if ok:
        x, y, w, h = [int(v) for v in box]
        x2 = x + w
        y2 = y + h
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 3)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"Task 3 CSRT Target {candidate_id}: {target_class}"
        cv2.putText(frame, label, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        trajectory_rows.append([
            frame_idx, 1, candidate_id, target_class, x, y, x2, y2, cx, cy
        ])

    else:
        cv2.putText(frame, "Target lost by CSRT tracker",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        trajectory_rows.append([
            frame_idx, 0, candidate_id, target_class, "", "", "", "", "", ""
        ])

    cv2.putText(frame, "Task 3 Single-Object Tracking using CSRT",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

with open(out_csv, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow([
        "frame", "found", "candidate_id", "class",
        "x1", "y1", "x2", "y2", "center_x", "center_y"
    ])
    writer_csv.writerows(trajectory_rows)

print("\nSaved CSRT Task 3 video:", out_video)
print("Saved CSRT trajectory CSV:", out_csv)
