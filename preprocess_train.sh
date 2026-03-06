#!/bin/bash
# Train nnU-Net 2D model for Dataset 999 fold 0. Max 5 folds can be run

# 1️⃣ Preprocess dataset for nnU-Net
nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity

# 2️⃣ Train the 2D nnU-Net model on fold 0
nnUNetv2_train 999 2d 0 --npz --c
