# YOLO11n Test Model - 3 Epochs

This folder stores the first small YOLO training test on the VisDrone dataset.

## Purpose

The goal of this run was not to get the best accuracy yet. The goal was to confirm that the full training pipeline works:

- VisDrone dataset can be read correctly
- YOLO labels are working
- GPU training works
- Validation runs successfully
- Model weights and metrics are saved

## Training Setup

- Model: YOLO11n
- Dataset: VisDrone DET train/val
- Epochs: 3
- Image size: 416
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU

## Main Results

- Precision: 0.454
- Recall: 0.118
- mAP50: 0.079
- mAP50-95: 0.0404

## Notes

This was only a quick test run. The model already improved compared to the pretrained validation check, which shows that it is learning from VisDrone.

The next step is to run a longer YOLO training experiment with more epochs.
