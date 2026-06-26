# Drone Vision Detection

This project is part of my internship work focused on computer vision for drone-based systems. The main goal is to build a clear and reusable object detection pipeline using the VisDrone dataset. The first stage of the project focuses on training and testing a YOLO model on drone-view images. After the image pipeline works, the next step is to test detection on drone videos and prepare the work so it can later connect to federated learning experiments.

I chose the name **Drone Vision Detection** because the project may start with YOLO, but it should not be limited only to YOLO. In the future, this same project structure can also be used to test other object detection models such as DETR, Faster R-CNN, RT-DETR, or other models that may be useful for drone perception.

## Project Goal

The goal of this project is to train, test, and evaluate object detection models using only the VisDrone dataset.

The first model I will work with is a YOLO model because YOLO is commonly used for real-time object detection and is a good starting point for drone-view images. The project will begin with image-based object detection, then move to video inference, and later explore how this work can be extended toward federated learning.

In simple terms, I want to answer these questions:

- Can a YOLO model detect objects accurately in drone-view images?
- How well does the model perform on small objects, crowded scenes, and aerial viewpoints?
- What are the main challenges when detecting people, vehicles, and other objects from drone data?
- How can this computer vision pipeline later support federated learning for drone swarm simulation?

## Dataset

This project will use the **VisDrone dataset** only.

VisDrone is a drone-view dataset that contains images and videos captured from UAVs/drones. It includes objects such as pedestrians, people, bicycles, cars, vans, trucks, buses, motors, and other vehicle types.

For this project, VisDrone will be used for:

- Training
- Testing
- Evaluation
- Image detection
- Video detection

No other dataset will be used for training, testing, or evaluation unless the team later decides otherwise.

## Why VisDrone?

VisDrone is useful for this project because it matches the drone simulation and drone swarm direction of the team. Normal object detection datasets are often based on regular street-view or ground-level images, but drone images are different. Objects can appear smaller, farther away, crowded, or viewed from above.

This makes VisDrone a better fit for testing object detection in aerial scenes.

## Main Project Steps

The project will be completed in stages.

### Step 1: Environment Setup

The first step is to set up the working environment using Conda, PyTorch, CUDA, and Ultralytics.

This is important because training object detection models requires many libraries, and using a clean environment helps avoid dependency problems.

### Step 2: Understand the Basics

Before training the model, I will review the important concepts needed for this project, including:

- CNNs
- PyTorch
- CUDA
- Conda environments
- Loss functions
- MSE
- Radians
- Object detection metrics

The purpose is not to become an expert in everything at once, but to understand enough to train, test, debug, and explain the model properly.

### Step 3: Prepare the VisDrone Dataset

The VisDrone annotations need to be prepared for YOLO training. YOLO expects labels in a specific format, so one major part of the project is converting the VisDrone annotation format into YOLO format.

YOLO label format looks like this:

class_id x_center y_center width height

The bounding box values must be normalized between 0 and 1.

### Step 4: Train the YOLO Baseline

After the dataset is prepared, I will train a YOLO model on VisDrone images.

Since my GPU has limited memory, I will begin with a lightweight YOLO model, such as YOLO11n, and use small batch sizes and image sizes that my computer can handle.

The first goal is not to get perfect accuracy immediately. The first goal is to build a working baseline that trains successfully and produces valid results.

### Step 5: Evaluate the Model

After training, I will evaluate the model using object detection metrics such as:

- Precision
- Recall
- mAP50
- mAP50-95
- Inference speed
- Visual detection quality

These metrics will help show how well the model performs on VisDrone data.

### Step 6: Test on Images

The trained model will be tested on VisDrone images. I will save examples of good detections and failure cases.

This is important because metrics alone do not show the full story. Looking at the output images helps explain where the model works well and where it struggles.

Examples of possible challenges include:

- Small objects
- Crowded scenes
- Overlapping vehicles or pedestrians
- Low-resolution objects
- Similar object classes
- Drone-view angles

### Step 7: Test on Videos

After the image pipeline works, I will run the trained model on VisDrone videos.

The video step will help test whether the model can perform detection frame by frame and whether the detections look stable over time.

### Step 8: Federated Learning Extension

After the basic computer vision pipeline is working, the next step is to think about how this can connect to federated learning.

The idea is that the VisDrone dataset could be split into multiple simulated clients. Each client could represent a drone, a different location, or a different type of scene. Each client would train locally, and later the model updates could be combined using a federated learning method such as FedAvg.

This would connect the object detection work to the larger drone swarm and federated learning direction of the team.

## Hardware

This project is being developed on my laptop.

Hardware:

- NVIDIA GeForce GTX 1650, 4 GB
- AMD Radeon integrated graphics
- Ubuntu Linux

For deep learning training, the NVIDIA GPU is the important one because CUDA works with NVIDIA GPUs. Since the GTX 1650 has only 4 GB of GPU memory, I will need to use smaller models, smaller image sizes, and smaller batch sizes.

## Software and Tools

The main tools planned for this project are:

- Ubuntu Linux
- Conda
- Python
- PyTorch
- CUDA
- Ultralytics YOLO
- OpenCV
- VisDrone dataset

More tools may be added later if needed for evaluation, visualization, or federated learning.

## Current Folder Structure

drone-vision-detection/
├── data/
├── models/
├── notes/
│   ├── cnn_notes.md
│   ├── cuda_notes.md
│   ├── daily_log.md
│   ├── pytorch_notes.md
│   └── questions_for_team.md
├── README.md
├── results/
│   ├── images/
│   ├── metrics/
│   └── videos/
├── runs/
└── scripts/
    ├── convert_visdrone_to_yolo.py
    ├── evaluate_model.py
    ├── predict_images.py
    ├── predict_video.py
    └── train_yolo.py

## Folder Explanation

### data/

This folder will store the VisDrone dataset and any converted dataset folders.

### scripts/

This folder will store Python scripts for dataset conversion, training, prediction, video inference, and evaluation.

### results/

This folder will store the outputs of the project, including prediction images, video results, and metric tables.

### notes/

This folder will store my learning notes, daily progress logs, and questions for the team.

### runs/

This folder will store training outputs generated by YOLO.

### models/

This folder will be used to store trained model weights or model-related files.

## Planned Deliverables

By the end of the main technical work, I want to have:

- A clean project structure
- A working Python/Conda environment
- VisDrone dataset organized
- VisDrone annotations converted into YOLO format
- A trained YOLO baseline model
- Evaluation metrics
- Image detection results
- Video detection results
- Notes about challenges and failure cases
- A short federated learning extension plan
- A final summary that can be used for presentation

## Challenges I Expect

Some challenges I expect during this project are:

- Setting up CUDA and PyTorch correctly
- Working with limited GPU memory
- Understanding the VisDrone annotation format
- Converting annotations correctly
- Detecting very small objects in drone images
- Handling crowded scenes
- Evaluating the model properly
- Explaining how the computer vision work connects to federated learning

## Long-Term Direction

The long-term direction of this project is to support drone swarm simulation and federated learning research. The object detection pipeline can become the computer vision baseline that later allows multiple drone agents or simulated clients to train and evaluate models without using a single centralized dataset.

This project starts with a simple goal: make object detection work properly on VisDrone. From there, it can grow into a stronger research pipeline for drone perception and federated learning.
