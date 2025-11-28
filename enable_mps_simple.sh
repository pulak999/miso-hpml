#!/bin/bash

# Enable MPS for a GPU without MIG
# Usage: ./enable_mps_simple.sh <gpu_id>

GPU_ID=$1

# Set GPU to EXCLUSIVE_PROCESS mode
sudo nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS

# Create MPS directories
mkdir -p /tmp/mps_log/nvidia-mps$GPU_ID
mkdir -p /tmp/mps_log/nvidia-log$GPU_ID

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_log/nvidia-mps$GPU_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log/nvidia-log$GPU_ID

# Start MPS daemon
nvidia-cuda-mps-control -d

echo "MPS enabled for GPU $GPU_ID"

