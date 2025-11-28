#!/bin/bash

# Disable MPS for a GPU
# Usage: ./disable_mps_simple.sh <gpu_id>

GPU_ID=$1

# Quit MPS control
echo quit | nvidia-cuda-mps-control

# Reset GPU to default compute mode
sudo nvidia-smi -i $GPU_ID -c DEFAULT

echo "MPS disabled for GPU $GPU_ID"

