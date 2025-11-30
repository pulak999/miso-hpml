#!/bin/bash

# Try to quit MPS control daemon (ignore error if not running)
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Kill any remaining MPS processes
pkill -9 nvidia-cuda-mps 2>/dev/null || true

# Reset GPU compute mode to default
sudo nvidia-smi -i 0,1 -c DEFAULT
