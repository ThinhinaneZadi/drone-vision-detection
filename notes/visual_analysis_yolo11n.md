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

## Confidence Threshold Test

I tested the 20-epoch YOLO11n model on one validation image using a lower confidence threshold.

Original prediction threshold:

- conf = 0.25

Lower threshold test:

- conf = 0.15

With the lower threshold, the model detected more pedestrians in the image. This shows that the model is sometimes detecting small pedestrians, but with low confidence.

The tradeoff is that lower confidence may increase the number of detections, but some detections may be less reliable.

This suggests that the model needs more training or improved settings to become more confident on small objects.

## Crowded Scene Confidence Test

I also tested a crowded traffic image using a lower confidence threshold of 0.15.

The model detected many more cars, along with a few pedestrians and one motor. However, the visual result became very cluttered because many labels overlapped in the dense traffic area.

This shows that lowering the confidence threshold can increase detections, but it does not always make the output better. In crowded scenes, it may make predictions harder to read and less clean.

Overall, lower confidence is useful for analysis, but the default threshold may be cleaner for presentation.

## Confidence Threshold Conclusion

After testing different confidence thresholds, I observed that all thresholds still missed objects, especially small pedestrians, motors, and objects in crowded scenes.

The lower threshold showed more detections, but the output became more cluttered. The higher threshold looked cleaner, but it missed even more small objects.

This means the main problem is not only the confidence threshold. The model needs stronger training or improved settings to detect small drone-view objects better.

## Continued Model and IoU Test

After continuing training for 30 more epochs, the model detected more pedestrians compared to the earlier 20-epoch baseline. However, some detections became overlapping and visually cluttered.

I tested a lower IoU threshold of 0.40 during prediction. This helped reduce overlapping boxes and made the output cleaner.

Even with the cleaner IoU setting, the model still missed some people, especially small or distant pedestrians. This shows that longer training and NMS tuning can improve results, but small-object detection remains a challenge in drone-view images.

## Video Pipeline Test

I created a short test video from 100 VisDrone validation images using FFmpeg. This was not a real drone video sequence, but an image-sequence video used to verify that the trained YOLO model can run inference on video input.

The YOLO11n continued model successfully processed the test video and generated an output video with detection boxes.

This confirms that the video inference pipeline works. The next step is to test the model on real VisDrone video data or another actual drone video sequence.

## Real VisDrone Video Sequence Test

I tested the continued YOLO11n model on a real VisDrone video sequence from the VisDrone2019-VID test-dev split.

The selected sequence was:

uav0000297_00000_v

The sequence contained 146 frames. I converted the image sequence into a short MP4 video using FFmpeg, then ran YOLO inference on the video.

The model successfully generated an output video with detection boxes. Visual inspection showed that the model can run on real VisDrone video data and produce useful detections.

This confirms that the project pipeline now supports both image inference and real video-sequence inference.

## External Drone Video Test

I also tested the continued YOLO11n model on an external drone video showing a road/bridge scene with cars, pedestrians, and motors.

The model successfully processed the external video and detected several relevant classes, including cars, pedestrians, and motors.

The result shows that the VisDrone-trained model can generalize to an outside drone-style video. However, the output still became cluttered in crowded areas, and some small objects were missed or confused.

This suggests that the model is useful for general drone-view detection, but further training and tuning are still needed for cleaner and more reliable results.

## Batch Size Choice

For training, I used a batch size of 1.

Batch size means how many images the model processes before updating its weights. A larger batch size can sometimes make training more stable, but it also uses more GPU memory.

I used batch=1 because my laptop GPU is an NVIDIA GTX 1650 with about 4GB of VRAM. Since VisDrone images are large and drone-view object detection can be memory intensive, using a larger batch size could cause CUDA out-of-memory errors.

Using batch=1 allowed training to run safely and made it possible to test larger image sizes, such as imgsz=512, without exceeding the GPU memory limit.
