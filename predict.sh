#!/bin/bash

# Predict using trained weights and save probabilities
nnUNetv2_predict \
    -i /app/nnUNet_raw/Dataset999_SPECT/imagesTs \   # input images
    -o /app/nnUNet_results/Dataset999_SPECT/predictions \   # output folder
    -tr nnUNetTrainerV2 \   # trainer class (uses the trained weights in nnUNet_results)
    -m 2d \   # 2D nnU-Net model
    -f 0 \    # fold 0
    --save_probabilities   # save probability maps in addition to masks
