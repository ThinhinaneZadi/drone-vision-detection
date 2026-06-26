# YOLO11n Baseline - 20 Epochs

This folder stores the first real YOLO11n baseline trained on the VisDrone detection dataset.

## Purpose

The goal of this run was to train a longer baseline model after confirming that the pipeline worked with the 3-epoch test run.

## Training Setup

- Model: YOLO11n
- Dataset: VisDrone DET train/val
- Epochs: 20
- Image size: 416
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU
- Training data: 6,471 images
- Validation data: 548 images

## Main Validation Results

- Precision: 0.249
- Recall: 0.172
- mAP50: 0.125
- mAP50-95: 0.0653

## Best Performing Class

The model performed best on cars.

Car results:

- Precision: 0.373
- Recall: 0.634
- mAP50: 0.527
- mAP50-95: 0.306

## Observation

The model improved compared to the 3-epoch test run. It learned to detect cars better than other classes, which makes sense because cars are common and more visually clear in drone-view images.

Smaller or more difficult classes, such as bicycles and awning-tricycles, still performed weakly. This is expected because VisDrone images contain many small, crowded, and partially occluded objects.

## Notes

This run is the first usable baseline for the project. The next step is to run predictions using this trained model, save visual examples, and compare them with the earlier 3-epoch model.

## Visual Prediction Check

After training, I ran prediction on all 548 validation images.

Prediction output folder:

results/images/yolo11n_20epochs_predictions

### Visual Observation

The 20-epoch model detects more objects than the 3-epoch test model. It performs best on cars and also begins detecting pedestrians, people, and some motors.

### Strengths

- Cars are detected clearly in many images.
- The model shows improvement compared to the 3-epoch test.
- Some pedestrians and motors are detected in drone-view scenes.

### Weaknesses

- Crowded scenes create overlapping labels.
- Small objects are still difficult.
- Some objects are missed or classified incorrectly.
- Classes like bicycle, tricycle, and awning-tricycle are still weak.

### Interpretation

This model is a usable YOLO11n baseline for VisDrone object detection. It proves that the pipeline works and that YOLO is learning from the VisDrone dataset, but more training or stronger model settings may be needed for better small-object performance.
