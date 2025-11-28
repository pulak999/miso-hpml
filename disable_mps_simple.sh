#!/bin/bash
GPU_ID=$1
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i $GPU_ID -c DEFAULT
echo "MPS disabled for GPU $GPU_ID"
