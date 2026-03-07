nnUNetv2_predict \
    -i /app/nnUNet_raw/Dataset999_SPECT/imagesTs \
    -o /app/nnUNet_results/Dataset999_SPECT/predictions \
    -d 999 \
    -c 2d \
    -f 0 \
    -chk checkpoint_best.pth \
    --save_probabilities -device cpu
    
