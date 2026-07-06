
## YOLO11s Continued Training Result

Experiment:
- Model: YOLO11s
- Starting weights: models/yolo11s_img512_test/best.pt
- Image size: 512
- Continued training: 20 epochs
- Batch size: 1

Validation result:
- Images: 548
- Instances: 38,759
- Precision: 0.390
- Recall: 0.276
- mAP50: 0.247
- mAP50-95: 0.139

Main class observation:
- Car performed best:
  - Precision: 0.541
  - Recall: 0.736
  - mAP50: 0.687
  - mAP50-95: 0.442

Conclusion:
Continuing YOLO11s training improved all main metrics compared to the 5-epoch YOLO11s test. The model is improving, but small and crowded object classes remain difficult.

## YOLO11s Frozen-Head Training Result

Experiment:
- Model: YOLO11s
- Training type: Frozen-head transfer learning
- Frozen layers: 0–22
- Trainable layer: 23 Detect head
- Image size: 512
- Epochs: 20
- Batch size: 1
- Validation images: 548
- Validation instances: 38,759

Validation result:
- Precision: 0.283
- Recall: 0.196
- mAP50: 0.153
- mAP50-95: 0.0833

Comparison:
- Full fine-tuning YOLO11s best:
  - Precision: 0.404
  - Recall: 0.286
  - mAP50: 0.261
  - mAP50-95: 0.147

Interpretation:
The frozen-head model is lower than the full fine-tuning model after 20 epochs. This suggests that only retraining the Detect head may not be enough for VisDrone, and earlier feature layers may need some adaptation to drone-view imagery. The immediate next step is to continue frozen-head training for more epochs and check whether the validation metrics continue improving or plateau.
