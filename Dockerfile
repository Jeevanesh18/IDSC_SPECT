# Start from a base image that already has PyTorch and CUDA 
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime 
# Make python print logs immediately 
ENV PYTHONUNBUFFERED=1 
# Install the Python libraries we need 
RUN pip install --no-cache-dir nnunetv2 simpleitk nibabel 
# Set the working directory inside the container 
WORKDIR /app 
# Create the folders nnU-Net expects 
RUN mkdir -p /app/nnUNet_raw \
/app/nnUNet_preprocessed \
/app/nnUNet_results 
# Tell nnU-Net where those folders are 
ENV nnUNet_raw=/app/nnUNet_raw 
ENV nnUNet_preprocessed=/app/nnUNet_preprocessed 
ENV nnUNet_results=/app/nnUNet_results 
# Copy your repository files into the container 
COPY . /app 
# Default command when the container starts 
CMD ["/bin/bash"]
