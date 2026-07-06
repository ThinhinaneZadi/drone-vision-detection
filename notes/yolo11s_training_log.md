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
