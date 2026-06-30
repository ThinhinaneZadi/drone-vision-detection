# YOLO11n Continued Training - 30 More Epochs

This experiment continued training from the previous YOLO11n 20-epoch baseline model.

## Settings

- Starting model: YOLO11n 20-epoch baseline best.pt
- Additional epochs: 30
- Approximate total training: 50 epochs
- Image size: 416
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU
- Dataset: VisDrone2019-DET train/val

## Results

Final validation results:

- Precision: 0.286
- Recall: 0.194
- mAP50: 0.154
- mAP50-95: 0.0808

## Comparison to Previous Baseline

Previous 20-epoch baseline:

- mAP50: 0.125
- mAP50-95: 0.0653

Continued training improved the model:

- mAP50 improved from 0.125 to 0.154
- mAP50-95 improved from 0.0653 to 0.0808

## Main Observation

Training longer improved the model overall, but small objects such as pedestrians, motors, bicycles, and crowded-scene objects still remain challenging.
