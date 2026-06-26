# Day 1 Progress Summary

## Main Focus

Today I focused on setting up the computer vision pipeline for the VisDrone object detection task. The goal was to create a clean project structure, prepare the environment, download and organize the dataset, convert annotations into YOLO format, and run initial YOLO training experiments.

## Project Setup

I created the main project folder:

drone-vision-detection

The project is organized into folders for data, scripts, results, notes, model files, and training outputs. This structure makes it easier to keep the dataset, code, metrics, predictions, and documentation separated.

## Environment Setup

I installed Miniconda and created a dedicated Conda environment called:

drone-vision

Inside this environment, I installed PyTorch with CUDA support and Ultralytics YOLO. I verified that PyTorch can access the NVIDIA GPU.

System details:

- Python: 3.10.20
- PyTorch: 2.12.1+cu126
- CUDA available in PyTorch: True
- GPU: NVIDIA GeForce GTX 1650
- GPU memory: about 4 GB
- Ultralytics YOLO installed successfully

This confirms that the environment is ready for YOLO training and inference.

## Dataset Preparation

I downloaded the VisDrone2019 detection image dataset using only the training and validation splits.

Dataset counts:

- Training images: 6,471
- Training annotations: 6,471
- Validation images: 548
- Validation annotations: 548

I deleted the downloaded zip files after extraction to save disk space.

## Annotation Conversion

The original VisDrone annotation format uses:

bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

YOLO requires:

class_id x_center_normalized y_center_normalized width_normalized height_normalized

I created a Python script called:

scripts/convert_visdrone_to_yolo.py

This script converts the VisDrone annotation files into YOLO label format. It also skips ignored regions and the vague "others" class.

After running the script, I verified that every image has a matching YOLO label file:

- Training YOLO labels: 6,471
- Validation YOLO labels: 548

## YOLO Dataset Configuration

I created a dataset configuration file:

visdrone.yaml

This file tells YOLO where the training and validation images are located and defines the 10 object classes used from VisDrone:

- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor

## Initial YOLO Validation Check

Before training, I ran a validation check using the pretrained YOLO11n model to make sure YOLO could read the dataset correctly.

The score was very low, which was expected because the model had not been trained on VisDrone yet. The purpose of this step was only to confirm that the dataset paths and labels were working.

## Training Experiments

### Experiment 1: YOLO11n Test Run

I trained YOLO11n for 3 epochs as a quick test to confirm that the full training pipeline works.

Settings:

- Model: YOLO11n
- Epochs: 3
- Image size: 416
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU

Results:

- Precision: 0.454
- Recall: 0.118
- mAP50: 0.079
- mAP50-95: 0.0404

This confirmed that the model was learning from the VisDrone dataset.

### Experiment 2: YOLO11n Baseline Run

I then trained YOLO11n for 20 epochs as the first real baseline.

Settings:

- Model: YOLO11n
- Epochs: 20
- Image size: 416
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU

Results:

- Precision: 0.249
- Recall: 0.172
- mAP50: 0.125
- mAP50-95: 0.0653

The 20-epoch model improved compared to the 3-epoch model in recall, mAP50, and mAP50-95.

## Visual Prediction Results

I ran predictions on all 548 validation images using both the 3-epoch test model and the 20-epoch baseline model.

The 20-epoch model showed better visual performance. It detected more objects, especially cars, pedestrians, people, and some motors.

Main observations:

- Cars are detected the best.
- Pedestrians are detected in several images.
- Some motors are detected.
- Crowded scenes create overlapping labels.
- Small objects are still difficult.
- Classes like bicycle, tricycle, and awning-tricycle remain weak.

## Saved Outputs

Important outputs were saved into organized folders.

Model folders:

- models/yolo11n_test_3epochs
- models/yolo11n_20epochs

Prediction folders:

- results/images/yolo11n_test_predictions
- results/images/yolo11n_20epochs_predictions

Metrics folder:

- results/metrics/yolo11n_3epochs_results.csv
- results/metrics/yolo11n_20epochs_results.csv
- results/metrics/yolo11n_metrics_summary.md

## Main Takeaway

Today I successfully built the first working VisDrone object detection pipeline. The dataset is organized, annotations are converted, YOLO training works on the GPU, and the first baseline model has been trained and evaluated.

The 20-epoch YOLO11n model is a usable baseline. It performs best on cars, but smaller and crowded objects remain challenging, which is expected for drone-view object detection.

## Next Steps

The next steps are:

1. Clean and review selected good and bad prediction examples.
2. Prepare a short visual analysis of the baseline model.
3. Decide whether to run another experiment with different settings, such as higher image size or longer training.
4. Start preparing for video inference after the image baseline is documented.
