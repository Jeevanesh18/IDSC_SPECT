#!/bin/bash

# Predict using trained weights and save probabilities
nnUNetv2_predict \
    -i /app/nnUNet_raw/Dataset999_SPECT/imagesTs \
    -o /app/nnUNet_results/Dataset999_SPECT/predictions \
    -d 999 \
    -c 2d \
    -tr nnUNetTrainerV2 \
    -f 0 \
    --save_probabilities \
