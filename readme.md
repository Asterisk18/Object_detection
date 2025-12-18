# Robust Object Detection, Tracking, and Classification System

A real-time computer vision system for object detection, multi-object tracking, and classification using classical image processing and a custom-trained machine learning model. The project focuses on data ownership, careful tuning of vision pipelines, and efficient CPU-only execution.

---

## Installation and Usage

```bash
git clone <repository-url>
cd <repository-name>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

# Project Overview

This project implements an end-to-end object perception pipeline that processes live webcam input to detect objects, track them across frames with persistent identities, and assign semantic labels. All components—data collection, detection, tracking, and classification—are implemented from scratch, allowing fine-grained control over system behavior.

A central focus of the project is dataset creation and labeling, highlighting the importance of data quality and annotation consistency in building reliable machine learning systems.

# Dataset and Model

Built a custom dataset of 200+ manually captured and labeled images

Iteratively refined labels after analyzing misclassifications

Trained a HOG + SVM classifier using scikit-learn

Supported classes: bottle, book, background, unknown

Observed that dataset refinement had a larger impact on performance than model changes

# Key Implementation Details

Detection pipeline using CLAHE, Gaussian smoothing, and adaptive Canny edge detection, tuned to handle lighting variation

Contour filtering and morphological operations to reduce noise and handle partial occlusion

Custom multi-object tracker using IoU-based assignment and Hungarian matching

Distance-aware Non-Max Suppression to prevent merging of nearby objects

Confidence-based classification with fallback to unknown to reduce false positives