# face-swap
A lightweight, efficient face-swapping engine designed for low-resource systems.

## Overview
**face-swap** is an experimental project done as a proof of concept to showcase how face swapping and deepfakes can be achieved *without* deep learning, heavy GPUs or massive compute budgets. Instead of neural networks, it employs the use of classical computer vision techniques to:

- Detetct facial landmarks
- Extract expressive features
- Align + blend them onto a target face
- Adjust tone + lighting for a natural result

## Features
-  **Facial landmark detection** – takes landmarks from the source image and the target image
-  **Feature extraction + alignment**
-  **skin-tone and colour blending**
-  **Static image face-swapping**

## Tech Stack
- **Python3**
- **OpenCV** – image processing and warping
- **Mediapipe** – facial landmarks
- **NumPy** – numerical operations
- **Pillow** – blending and colour processing

## Architechture
face-swap/
|
|–– swapper.py
|
|––shape_predictor_68_face_landmark.dat
| # Required facial landmarking model

