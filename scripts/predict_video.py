from ultralytics import YOLO


# This script runs YOLO object detection on a video file.
# It can be used later for real drone videos or simulation videos.


def main():
    # Load the trained YOLO model.
    # This is the continued model trained after the 20-epoch baseline.
    model = YOLO("models/yolo11n_continue_30epochs/best.pt")

    # Path to the input video.
    # Replace this later with a real drone video path.
    video_path = "results/videos/input_video.mp4"

    # Run prediction on the video.
    # conf controls the minimum confidence needed to show a detection.
    # iou controls how strongly overlapping boxes are removed.
    model.predict(
        source=video_path,
        imgsz=416,
        conf=0.25,
        iou=0.40,
        device=0,
        project="results/videos",
        name="video_prediction",
        save=True,
    )


# This makes sure the script runs only when we execute this file directly.
if __name__ == "__main__":
    main()
