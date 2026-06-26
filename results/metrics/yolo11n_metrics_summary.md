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
