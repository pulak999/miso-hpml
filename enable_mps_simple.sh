#!/bin/bash

# Enable MPS for a GPU without MIG
# Usage: ./enable_mps_simple.sh <gpu_id>

GPU_ID=$1
USER=$(whoami)
HOSTNAME=$(hostname)

# Set GPU to EXCLUSIVE_PROCESS mode
sudo nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS

# Create MPS directories (matching training script paths)
mkdir -p /scratch/$USER/mps_log/nvidia-mps-$HOSTNAME/$GPU_ID
mkdir -p /scratch/$USER/mps_log/nvidia-log-$HOSTNAME/$GPU_ID

# Set environment variables (matching training script paths)
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_MPS_PIPE_DIRECTORY=/scratch/$USER/mps_log/nvidia-mps-$HOSTNAME/$GPU_ID
export CUDA_MPS_LOG_DIRECTORY=/scratch/$USER/mps_log/nvidia-log-$HOSTNAME/$GPU_ID

# Start MPS daemon
nvidia-cuda-mps-control -d

echo "MPS enabled for GPU $GPU_ID"

