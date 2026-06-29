# Visual Analysis: YOLO11n Baseline on VisDrone

## Purpose

This note summarizes the visual performance of the YOLO11n 20-epoch baseline model on the VisDrone validation images.

## What the Model Does Well

The model detects cars better than other classes. Cars are usually larger, clearer, and more repeated in the dataset, so the model learns them more easily.

The model also detects some pedestrians, people, vans, trucks, and motors, especially when they are clear and not too small.

## Main Limitations

The model struggles with small objects, crowded scenes, and low-light images. In many drone-view images, pedestrians and motors are very small, so the model often misses them.

Some predictions are also messy in crowded scenes, with overlapping labels or incomplete detections.

## Observed Failure Cases

The main failure cases are:

- Missed pedestrians in crowded areas
- Missed small vehicles and motors
- Overlapping labels in dense traffic scenes
- Weak detection in dark or low-light images
- Occasional wrong class prediction

## Main Takeaway

The YOLO11n 20-epoch model is a good first baseline. It works best for cars but needs improvement for small objects and crowded drone-view scenes.

## Next Improvement Ideas

Possible next steps:

- Try a lower confidence threshold to see if the model can detect more small objects
- Train for more epochs
- Try a larger image size if GPU memory allows
- Compare with another YOLO model size later
