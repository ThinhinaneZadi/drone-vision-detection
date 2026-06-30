from ultralytics import YOLO


# This script runs YOLO object detection on VisDrone validation images.
# It is useful for visual checking after training a model.


def main():
    # Load the trained model.
    # This is the continued model after additional training.
    model = YOLO("models/yolo11n_continue_30epochs/best.pt")

    # Path to the validation images.
    image_folder = "data/VisDrone-DET/VisDrone2019-DET-val/images"

    # Run prediction on the images.
    # conf controls the minimum confidence needed to show a detection.
    # iou helps reduce overlapping boxes.
    model.predict(
        source=image_folder,
        imgsz=416,
        conf=0.25,
        iou=0.40,
        device=0,
        project="results/images",
        name="yolo11n_continue_predictions",
        save=True,
    )


# This makes sure the script runs only when we execute this file directly.
if __name__ == "__main__":
    main()
