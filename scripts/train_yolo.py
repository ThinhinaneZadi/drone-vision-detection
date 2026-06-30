from ultralytics import YOLO


# This script trains a YOLO model on the VisDrone dataset.
# It is useful because it saves the training setup in one clear file.


def main():
    # Load a YOLO model.
    # For a new training run, use "yolo11n.pt".
    # To continue training from a saved model, use a path like:
    # "models/yolo11n_20epochs/best.pt"
    model = YOLO("yolo11n.pt")

    # Train the model on the VisDrone dataset.
    model.train(
        data="visdrone.yaml",
        epochs=20,
        imgsz=416,
        batch=1,
        device=0,
        project="runs/detect",
        name="visdrone_yolo11n_training",
    )


# This makes sure the script runs only when we execute this file directly.
if __name__ == "__main__":
    main()
