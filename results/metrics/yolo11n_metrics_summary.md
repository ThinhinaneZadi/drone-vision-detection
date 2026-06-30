# YOLO11n Metrics Summary

This file summarizes the YOLO11n experiments completed so far on the VisDrone detection dataset.

## Dataset

- Dataset: VisDrone2019-DET
- Training images: 6,471
- Validation images: 548
- Classes used: 10
- Ignored classes: ignored regions and others

## Experiment 1: YOLO11n Test Run

This was a short training run used to confirm that the pipeline works.

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 3 |
| Image size | 416 |
| Batch size | 1 |
| Device | NVIDIA GTX 1650 |

### Results

| Metric | Value |
|---|---:|
| Precision | 0.454 |
| Recall | 0.118 |
| mAP50 | 0.079 |
| mAP50-95 | 0.0404 |

## Experiment 2: YOLO11n Baseline

This was the first longer baseline training run.

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 20 |
| Image size | 416 |
| Batch size | 1 |
| Device | NVIDIA GTX 1650 |

### Results

| Metric | Value |
|---|---:|
| Precision | 0.249 |
| Recall | 0.172 |
| mAP50 | 0.125 |
| mAP50-95 | 0.0653 |

## Comparison

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n - 3 epochs | 0.454 | 0.118 | 0.079 | 0.0404 |
| YOLO11n - 20 epochs | 0.249 | 0.172 | 0.125 | 0.0653 |

## Interpretation

The 20-epoch model improved in recall, mAP50, and mAP50-95 compared to the 3-epoch test run. This means the model learned more from the VisDrone dataset after longer training.

The precision decreased, which means the model is making more detections but also producing more false positives. This is common when a model starts detecting more objects, especially in crowded drone-view scenes.

Overall, the 20-epoch model is a stronger baseline because it detects more objects and has better mAP scores.

## Main Observation

Cars are the strongest class so far. Smaller objects such as bicycles, tricycles, awning-tricycles, and some pedestrians remain difficult because VisDrone contains many small, crowded, and partially occluded objects.

## Experiment 3: YOLO11n Continued Training

This experiment continued training from the 20-epoch YOLO11n baseline for 30 additional epochs.

Settings:

- Starting model: YOLO11n 20-epoch baseline best.pt
- Additional epochs: 30
- Approximate total training: 50 epochs
- Image size: 416
- Batch size: 1
- Device: NVIDIA GTX 1650 GPU

Results:

- Precision: 0.286
- Recall: 0.194
- mAP50: 0.154
- mAP50-95: 0.0808

Comparison:

- 20-epoch baseline mAP50: 0.125
- Continued model mAP50: 0.154
- 20-epoch baseline mAP50-95: 0.0653
- Continued model mAP50-95: 0.0808

Main takeaway:

Continuing training improved the model overall, especially in recall and mAP. However, visual testing still showed missed small objects and some overlapping detections, so small-object detection remains the main challenge.
