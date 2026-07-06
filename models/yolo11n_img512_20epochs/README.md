# YOLO11n Image Size 512 Experiment

This experiment continued from the previous YOLO11n model and trained using a larger image size.

## Settings

- Starting model: YOLO11n continued model best.pt
- Epochs: 20
- Image size: 512
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU
- Dataset: VisDrone2019-DET train/val

## Results

Final validation results:

- Precision: 0.30779
- Recall: 0.23741
- mAP50: 0.19220
- mAP50-95: 0.10373

## Comparison to Previous Model

Previous continued model with image size 416:

- Precision: 0.286
- Recall: 0.194
- mAP50: 0.154
- mAP50-95: 0.0808

Image size 512 improved the model:

- Precision improved from 0.286 to 0.30779
- Recall improved from 0.194 to 0.23741
- mAP50 improved from 0.154 to 0.19220
- mAP50-95 improved from 0.0808 to 0.10373

## Main Observation

Increasing the image size helped the model detect more objects, especially smaller drone-view objects. The results are still limited because VisDrone is challenging and the model is YOLO11n running on a 4GB GPU, but this is the best experiment so far.
