# ☢️ SPECT-LV-Segmenter: Myocardial Perfusion Analysis

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

![alt text](https://img.shields.io/badge/Framework-nnU--Net_V2-red)

![alt text](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)

**Automated segmentation of the left ventricular wall from Myocardial Perfusion SPECT images using nnU-Net.**
Built for clinical reproducibility, interpretability, and robust performance on the PhysioNet MPS Dataset.

## 📋 Table of Contents
- 🎯 [The Mission](#the-mission)
- 📦 [Dataset & Preprocessing](#dataset--preprocessing)
- 🗂️ [Repository Structure](#repository-structure)
- 🐳 [Installation (Docker)](#installation-docker)
- 🚀 [Train from Scratch](#-option-1-train-from-scratch)
- 🧠 [Pretrained Inference](#-option-2-pretrained-inference)
- 📊 [Visualization & Insights](#visualization--insights)
- 🙏 [Credits](#credits)


## 🏥 The Mission
Cardiovascular diseases remain a leading cause of mortality globally. **Myocardial Perfusion SPECT (Single-Photon Emission Computed Tomography)** is a critical imaging modality for assessing blood flow to the heart muscle (left ventricle).

This repository provides an end-to-end pipeline to:

1. **Standardize Data**: Convert raw clinical DICOM data into NIfTI formats suitable for deep learning.
2. **Segment**: Automatically identify the Left Ventricle (LV) using nnU-Net, the state-of-the-art self-configuring framework for medical segmentation.
3. **Explain**: Provide interpretable outputs (Probability Heatmaps) to aid clinicians in trusting the model ("The Insights").

## 📦 Dataset & Preprocessing
Source: [PhysioNet Myocardial Perfusion SPECT (MPS) Database]([https://github.com](https://physionet.org/content/myocardial-perfusion-spect/get-zip/1.0.0/))


The dataset consists of 83 unique patients acquired using a CZT-based gamma camera (Discovery NM 530c, GE Healthcare). Two most important files in the zip file are DICOM and NlfTI.

Data Split Strategy
To ensure strict evaluation, the data is split based on the availability of ground-truth masks:

- **Training/Validation**: 100 images (Paired with expert-annotated masks).
- **Testing/Inference**: Images without provided masks (Used to demonstrate model generalization on unseen data).
  
The Pipeline (prepare_dataset.py)

We implemented a robust preprocessing script that:
Downloads the raw data automatically.
Pairs DICOM inputs with NIfTI masks.
Resamples geometry to fix resolution mismatches.
Formats the directory structure strictly for nnU-Net (Dataset ID: 999).
📂 Repository Structure
Ensure your local folder looks like this before starting:
code
Text
IDSC_SPECT/
├── Dockerfile                  # Environment definition
├── prepare_dataset.py          # ETL pipeline (DICOM -> NIfTI)
├── preprocess_train.sh         # Shell script to trigger nnU-Net training
├── predict.sh                  # Shell script for inference
├── visualization.py            # Generates PDFs with Heatmaps
├── README.md                   # This file
└── nnUNet_data/                # (Created automatically or mounted)
    ├── nnUNet_raw/
    ├── nnUNet_preprocessed/
    └── nnUNet_results/
🛠 Installation (Docker)
To guarantee reproducibility, we use Docker. This ensures you run the exact same environment (CUDA 11.7, PyTorch 2.0, nnU-Net V2) as used during development.
1. Create the data directories on your host machine:
code
Bash
mkdir -p nnUNet_data/nnUNet_raw
mkdir -p nnUNet_data/nnUNet_preprocessed
mkdir -p nnUNet_data/nnUNet_results
2. Build the Docker image:
code
Bash
docker build -t spect-segmenter .
3. Run the container:
Note: We mount the local folders to the container so data persists after the container stops.
code
Bash
docker run --gpus all -it \
  -v $(pwd)/nnUNet_data/nnUNet_raw:/app/nnUNet_raw \
  -v $(pwd)/nnUNet_data/nnUNet_preprocessed:/app/nnUNet_preprocessed \
  -v $(pwd)/nnUNet_data/nnUNet_results:/app/nnUNet_results \
  spect-segmenter
You are now inside the container terminal.
🚀 Option 1: Train from Scratch
Follow these steps to reproduce the training pipeline completely.
1. Clone the repo (inside the container):
code
Bash
git clone https://github.com/Jeevanesh18/IDSC_SPECT.git
cd IDSC_SPECT
2. Download and Preprocess Data:
This script downloads the zip from PhysioNet, converts DICOMs, and splits the data.
code
Bash
python prepare_dataset.py
3. Train the Model:
We use a 2D nnU-Net configuration. This script verifies integrity and trains Fold 0.
code
Bash
bash preprocess_train.sh
🧠 Option 2: Use Pretrained Model
If you do not have the time or GPU resources to train, you can use my pretrained weights.
1. Download Weights:
Download the nnUNet_results folder from Google Drive Link Here.
2. Setup:
Unzip the content and place it into your local nnUNet_data/nnUNet_results folder.
3. Run Inference:
Start the Docker container (as shown in Installation). Run the prediction script to generate segmentation masks for the test set.
code
Bash
bash predict.sh
🔍 Visualization & Insights
Medical AI must be explainable. We generate detailed PDF reports for every patient in the test set.
Generate Reports:
code
Bash
python visualization.py
Output:
Check nnUNet_results/Dataset999_SPECT/visualizations. Each PDF contains:
Raw SPECT: The original input slice.
Segmentation: The predicted LV wall.
Uncertainty Heatmap: A probability map showing the model's confidence levels.
Example Output:
A heatmap showing high confidence (Red) in the ventricular wall and low confidence (Blue) in the background.
🤝 Credits
Dataset: PhysioNet / GE Healthcare
Methodology: nnU-Net: Self-adapting Framework for U-Net-based Medical Image Segmentation
Author: Jeevanesh18
