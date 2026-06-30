# Project Workflow

This project uses YOLO for object detection on the VisDrone dataset.

## 1. Prepare the Dataset

Convert VisDrone annotations into YOLO format:

python scripts/convert_visdrone_to_yolo.py

This creates YOLO label files inside the dataset folders.

## 2. Train the Model

Train YOLO on the VisDrone dataset:

python scripts/train_yolo.py

The training settings are saved inside the script.

## 3. Evaluate the Model

Evaluate the trained model on the validation set:

python scripts/evaluate_model.py

This gives metrics such as precision, recall, mAP50, and mAP50-95.

## 4. Predict on Images

Run object detection on validation images:

python scripts/predict_images.py

This saves images with detection boxes.

## 5. Predict on Video

Run object detection on a video file:

python scripts/predict_video.py

The current video script expects an input video at:

results/videos/input_video.mp4

This path can be changed later when using a real drone or simulation video.
