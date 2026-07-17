# VisDrone Tracking Tasks Log

## Task 4: Multi-Object Tracking with YOLO11s + ByteTrack

### Goal

The goal of this experiment was to extend the video object detection pipeline toward multi-object tracking. Instead of only detecting objects independently in each frame, the tracker assigns object IDs so objects can be followed across frames.

### Dataset

Dataset used:

- VisDrone2019-VID test-dev
- Sequence used: `uav0000297_00000_v`

The VisDrone-VID sequence was already stored as ordered image frames:

```text
0000001.jpg
0000002.jpg
0000003.jpg
...
