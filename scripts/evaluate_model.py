from ultralytics import YOLO


# This script evaluates a trained YOLO model on the VisDrone validation set.
# It reports metrics such as precision, recall, mAP50, and mAP50-95.


def main():
    # Load the trained model.
    # This is the continued model after the 20-epoch baseline.
    model = YOLO("models/yolo11n_continue_30epochs/best.pt")

    # Evaluate the model on the validation split defined in visdrone.yaml.
    results = model.val(
        data="visdrone.yaml",
        imgsz=416,
        batch=1,
        device=0,
        project="results/metrics",
        name="evaluation_yolo11n_continue",
    )

    # Print a short summary.
    print("Evaluation finished.")
    print("Check the saved results folder for detailed metrics.")
    print(results)


# This makes sure the script runs only when we execute this file directly.
if __name__ == "__main__":
    main()
