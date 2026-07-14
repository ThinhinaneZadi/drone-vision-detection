# Colab Training Progress

## Current Focus

The project moved from local GTX 1650 experiments to Google Colab Tesla T4 in order to improve YOLO11s detection performance on VisDrone.

The goal is to improve real UAV object detection accuracy, especially Precision, while keeping the experiments organized and comparable.

## Completed Experiments

### Frozen-head and partial-freeze experiments

Earlier experiments tested how much of YOLO11s needs to be trainable for VisDrone adaptation.

Main finding:

- Training only the Detect head improved over zero-shot YOLO, but performance was limited.
- Training the last two layers improved slightly.
- Training layers 16–23 improved much more.
- Full fine-tuning gave the strongest result so far.

### Best Result So Far

Full fine-tuning YOLO11s on Colab:

- Model: YOLO11s
- Dataset: VisDrone2019-DET
- Image size: 640
- Batch size: 16
- Optimizer: SGD
- Learning rate: 0.01
- Cosine LR: True
- AMP: True
- Epochs: stopped at 95 due to early stopping
- Best epoch: 65

Validation result:

- Precision: 0.521
- Recall: 0.400
- mAP50: 0.388
- mAP50-95: 0.227

This became the strongest result so far and confirmed that full fine-tuning is much more effective than frozen-layer training for VisDrone.

## Ongoing Experiment

Resolution adaptation is now being tested.

Current experiment:

- Starting model: best full fine-tuned YOLO11s checkpoint
- Image size: 768
- Batch size: 16
- Optimizer: SGD
- Learning rate: 0.002
- Epochs: 40
- Goal: improve small-object detection while keeping batch size at least 16 for more stable gradient updates.

## Next Planned Experiments

1. Continue resolution adaptation.
2. Test higher image resolution if GPU memory allows.
3. Evaluate SAHI sliced inference for small-object detection.
4. Compare per-class precision and recall, especially for cars, buses, pedestrians, bicycles, and tricycles.
5. Explore YOLO11s-P2 if time allows.

## Research Direction

The project is moving toward stronger UAV object detection for drone-view imagery. The results show a clear progression:

zero-shot YOLO → frozen head → last two layers → partial freeze → full fine-tuning → higher-resolution adaptation.

