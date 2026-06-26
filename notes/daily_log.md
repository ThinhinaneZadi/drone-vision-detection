
## Environment Setup

### Completed
- Installed Miniconda
- Created Conda environment: drone-vision
- Installed PyTorch with CUDA support
- Confirmed CUDA is available in PyTorch
- Confirmed GPU is detected: NVIDIA GeForce GTX 1650
- Installed Ultralytics YOLO
- Ran yolo checks successfully

### System Details
- Python: 3.10.20
- PyTorch: 2.12.1+cu126
- CUDA used by PyTorch: 12.6
- GPU: NVIDIA GeForce GTX 1650
- GPU memory: about 4 GB

### Notes
The environment is ready for the next step: downloading and organizing the VisDrone dataset.

## Dataset YOLO Validation Check

### Completed
- Created visdrone.yaml for YOLO training.
- Confirmed YOLO can read the VisDrone validation images.
- Confirmed YOLO can read the converted YOLO label files.
- Ran a validation check using pretrained yolo11n.pt.
- Results were saved in runs/detect/val.

### Result
The validation command ran successfully on the GPU, which means the dataset paths and label format are working.

### Important Note
The mAP score was very low because the model has not been trained on VisDrone yet. This step was only to check that the dataset setup works before training.

### Next Step
Start a small YOLO training test on VisDrone.

## First YOLO Training Test

### Completed
- Ran a small YOLO training test on the VisDrone dataset.
- Used YOLO11n as the starting model.
- Trained for 3 epochs.
- Used image size 416 and batch size 1 because the GPU has 4 GB of VRAM.
- Training completed successfully on the NVIDIA GTX 1650 GPU.
- Validation also completed successfully.

### Training Command
yolo detect train model=yolo11n.pt data=visdrone.yaml epochs=3 imgsz=416 batch=1 device=0 project=runs name=visdrone_test

### Main Validation Result
- Precision: 0.454
- Recall: 0.118
- mAP50: 0.079
- mAP50-95: 0.0404

### Observation
The model improved compared to the pretrained validation check, which means it is learning from the VisDrone dataset. The car class performed the best so far, while smaller or less frequent classes like bicycle, tricycle, and awning-tricycle were still weak.

### Important Note
This was only a short test run to confirm that the training pipeline works. The next step is to run a longer training experiment with a cleaner save path.

## Visual Prediction Check

### Completed
- Ran prediction on the full VisDrone validation image set using the 3-epoch YOLO11n test model.
- Saved 548 prediction images in results/images/yolo11n_test_predictions.
- Opened a sample prediction image to visually inspect the model output.

### Observation
The model is already detecting some larger vehicles, especially cars. Some predictions have reasonable confidence scores, such as around 0.78 and 0.80. However, the model is still missing many small objects such as pedestrians, motorcycles, and crowded scene objects.

### Interpretation
This result is expected because the model was trained for only 3 epochs. The short training test confirms that the pipeline works, but a longer training run is needed to improve performance on smaller and harder objects.

## Visual Check - 20 Epoch YOLO11n Model

### Completed
- Ran prediction on the VisDrone validation images using the 20-epoch YOLO11n model.
- Saved 548 prediction images in results/images/yolo11n_20epochs_predictions.
- Visually inspected several prediction examples.

### Observation
The 20-epoch model performs better than the 3-epoch test model. It detects more objects and is no longer limited mostly to cars. The model is now detecting cars, pedestrians, people, and some motors.

### Strengths
- Cars are detected with stronger confidence.
- Pedestrians are detected in several drone-view scenes.
- The model shows clear improvement compared to the 3-epoch test run.

### Weaknesses
- Crowded scenes still create messy overlapping labels.
- Small objects are still difficult to detect.
- Some pedestrians and motorcycles are missed.
- Some classes remain weaker than cars.

### Interpretation
The 20-epoch YOLO11n model is a usable baseline. It shows that the model is learning from VisDrone, but drone-view object detection remains challenging because many objects are small, crowded, or partially occluded.
