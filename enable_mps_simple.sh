#!/bin/bash
GPU_ID=$1
sudo nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS
mkdir -p /tmp/mps_log/nvidia-mps$GPU_ID
mkdir -p /tmp/mps_log/nvidia-log$GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_log/nvidia-mps$GPU_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log/nvidia-log$GPU_ID
nvidia-cuda-mps-control -d
echo "MPS enabled for GPU $GPU_ID"
