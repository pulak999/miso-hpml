# Complete MPS Experiment Setup Guide

This guide provides step-by-step instructions for running MPS (Multi-Process Service) experiments with **macOS as the client/scheduler** and **Ubuntu as the GPU server**.

## Architecture Overview

```
┌─────────────────┐         TCP/IP          ┌──────────────────┐
│   macOS Client  │  ←──────────────────→   │  Ubuntu GPU      │
│   (Scheduler)   │   Port 10002            │  Server          │
│                 │                          │  (Experiments)   │
│ - Runs exp_mps  │                          │ - Runs gpu_server│
│ - Sends commands│                          │ - Executes jobs  │
│ - No GPU needed │                          │ - Has GPUs       │
└─────────────────┘                          └──────────────────┘
```

---

## Prerequisites

### macOS Client
- macOS with Python 3.7+ and conda/miniconda installed
- SSH access to Ubuntu server
- Network connectivity to Ubuntu server (port 10002)

### Ubuntu GPU Server
- Ubuntu 18.04, 20.04, 22.04, or later
- NVIDIA GPU (L4, A100, or any CUDA-compatible GPU)
- NVIDIA drivers installed (`nvidia-smi` should work)
- CUDA toolkit installed
- Python 3.7+ and conda/miniconda installed
- Sudo/root access for GPU configuration
- Port 10002 available for TCP communication

---

## Step 1: macOS Client/Scheduler Setup

### 1.1 Clone Repository on macOS

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

# If repository is not already there, clone it:
# git clone <repository-url> socc22-miso
# cd socc22-miso
```

### 1.2 Create Minimal Python Environment

The scheduler only needs to send TCP commands - no GPU or CUDA required.

**Note for Apple Silicon (M1/M2/M3) Macs**: Python 3.7 may not be available. Use Python 3.8 or 3.9 instead.

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

# Check available Python versions (especially important for Apple Silicon)
conda search python | grep "^python\s"

# Create minimal conda environment
# For Intel Macs or if Python 3.7 is available:
conda create -n miso_client python=3.7 -y

# For Apple Silicon Macs (M1/M2/M3), use Python 3.8 or 3.9:
# conda create -n miso_client python=3.8 -y
# OR
# conda create -n miso_client python=3.9 -y

conda activate miso_client

# Install only the Python dependencies needed for scheduling
pip install numpy pandas pyyaml psutil
```

**Alternative**: If you want to use the full environment file (for local testing):

```bash
# Create environment from file (will fail on CUDA packages - that's OK)
conda env create -f environment_mps.yml
conda activate tf2

# Remove CUDA packages (they won't work on macOS anyway)
pip uninstall torch torchvision torchaudio cudatoolkit cudnn -y 2>/dev/null || true
```

**Troubleshooting Python Version Issues:**

If you get `PackagesNotFoundError` for Python 3.7 on Apple Silicon:

1. **Use Python 3.8 or 3.9** (recommended):
   ```bash
   conda create -n miso_client python=3.8 -y
   # OR
   conda create -n miso_client python=3.9 -y
   ```

2. **Use conda-forge channel** (may have more Python versions):
   ```bash
   conda create -n miso_client -c conda-forge python=3.7 -y
   ```

3. **Use pyenv** (if you specifically need Python 3.7):
   ```bash
   # Install pyenv first: brew install pyenv
   pyenv install 3.7.17
   pyenv local 3.7.17
   python -m venv miso_client
   source miso_client/bin/activate
   ```

### 1.3 Verify Path Configuration

**✅ Good news!** The codebase already uses a configurable path system (`miso_config.py`), so no path fixing is needed. The files automatically detect the repository location.

**Just verify it works:**

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client  # or tf2
```

This should print your repository path. If it shows the correct path, you're all set!

**Note**: The `fix_paths_for_macos.py` script is no longer needed - all files already use `miso_config.py` for paths.

### 1.4 Set Repository Root Environment Variable

```bash
# On macOS
export MISO_REPO_ROOT=~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

# Test that the config system works
python -c "from miso_config import REPO_ROOT; print(f'Repo root: {REPO_ROOT}')"

# Add to ~/.zshrc or ~/.bash_profile for persistence
echo 'export MISO_REPO_ROOT=~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso' >> ~/.zshrc
source ~/.zshrc
```

### 1.5 Create Run Script for macOS (Optional but Recommended)

**Why this script?** The existing `run.py` has hardcoded physical node hostnames and doesn't support the `--mps_level` parameter or macOS client setup. This script provides:

- ✅ Configurable GPU server hostname (critical for macOS → Ubuntu setup)
- ✅ `--mps_level` parameter support
- ✅ Better error messages for client/server configuration
- ✅ Only runs MPS experiments (not all experiment types)

**Alternative**: You could modify `run.py` directly, but this script is cleaner for the macOS client use case.

Create `run_mps_only.py` in the repository root:

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

cat > run_mps_only.py << 'EOF'
#!/usr/bin/env python3
import argparse
import time
import socket
import os
from pathlib import Path
from exp_mps import MPS

parser = argparse.ArgumentParser(description='MPS-only experiment')
parser.add_argument('--arrival', type=int, default=60, help='inter-arrival period')
parser.add_argument('--num_job', type=int, default=30, help='total number of jobs')
parser.add_argument('--num_gpu', type=int, default=1, help='total number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--step', type=int, default=10, help='simulation step size')
parser.add_argument('--filler', action='store_true', default=False)
parser.add_argument('--flat_arrival', action='store_true', default=False)
parser.add_argument('--random_trace', action='store_true', default=False)
parser.add_argument('--mps_level', type=int, default=33, help='MPS thread percentage (33, 50, etc.)')
parser.add_argument('--gpu_server_host', type=str, default=None, help='GPU server hostname or IP')
args = parser.parse_args()

# Create logs directory
Path('logs/mps').mkdir(parents=True, exist_ok=True)

# Get GPU server hostname/IP
if args.gpu_server_host:
    physical_nodes = [args.gpu_server_host]
    print(f'Using GPU server: {args.gpu_server_host}')
else:
    # Try environment variable, then prompt
    gpu_host = os.environ.get('GPU_SERVER_HOST')
    if gpu_host:
        physical_nodes = [gpu_host]
        print(f'Using GPU server from env: {gpu_host}')
    else:
        print('ERROR: --gpu_server_host not specified and GPU_SERVER_HOST env var not set')
        print('Please specify: python run_mps_only.py --gpu_server_host <ubuntu-hostname-or-ip> ...')
        exit(1)

print('=' * 60)
print('Running MPS-Only Experiment')
print(f'Jobs: {args.num_job}, GPUs: {args.num_gpu}, MPS Level: {args.mps_level}')
print(f'GPU Server: {physical_nodes[0]}')
print('=' * 60)

mps_exp = MPS(args, physical_nodes)
mps_exp.run(args, mps_lvl=args.mps_level)

print('Experiment completed!')
print(f'Results saved in: logs/mps/')
EOF

chmod +x run_mps_only.py
```

### 1.6 Test Network Connectivity

```bash
# On macOS
# Replace <ubuntu-ip-or-hostname> with your Ubuntu server's IP or hostname
# First, get the IP from Ubuntu (see Step 2.1)

# Test connectivity
ping <ubuntu-ip-or-hostname>

# Test port 10002 (will fail until GPU server is running)
nc -zv <ubuntu-ip-or-hostname> 10002
```

**Note**: Port 10002 will only be open after you start the GPU server in Step 2.

---

## Step 2: Ubuntu GPU Server Setup

### 2.1 Initial SSH and Verification

```bash
# SSH into your Ubuntu GPU server
ssh -i your-key.pem ubuntu@your-instance-ip

# Verify GPU access
nvidia-smi

# Check CUDA MPS availability
which nvidia-cuda-mps-control

# Get hostname (you'll need this)
hostname
# Save this output - you'll use it in Step 1.6

# Get IP address
hostname -I
# Save this IP - you'll use it in Step 1.6
```

### 2.2 Clone and Prepare Repository

```bash
# On Ubuntu
# Create directory structure
mkdir -p ~/GIT
cd ~/GIT

# Clone the repository (or upload via scp from macOS)
# Option A: If you have git access
git clone <repository-url> socc22-miso

# Option B: Upload from macOS
# On macOS, run:
# scp -r -i your-key.pem ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso ubuntu@your-instance-ip:~/GIT/

cd ~/GIT/socc22-miso

# Verify structure
ls -la
```

### 2.3 Setup Python Environment

```bash
# On Ubuntu
cd ~/GIT/socc22-miso

# Create conda environment (use environment_mps.yml for MPS-only experiments)
conda env create -f environment_mps.yml

# Activate environment
conda activate tf2

# Verify installation
python -c "import torch; print(torch.__version__)"
nvidia-smi
```

### 2.4 Set Repository Root Environment Variable

```bash
# On Ubuntu
export MISO_REPO_ROOT=/home/$USER/GIT/socc22-miso

# Add to ~/.bashrc for persistence
echo 'export MISO_REPO_ROOT=/home/$USER/GIT/socc22-miso' >> ~/.bashrc
source ~/.bashrc

# Verify
python -c "from miso_config import REPO_ROOT; print(f'Repo root: {REPO_ROOT}')"
```

### 2.5 Create Device Mapping Configuration

**Why this file?** `gpu_server.py` **requires** `mig_device_autogen.json` at startup (line 43). It uses this file to:
- Map hostnames to GPU configurations
- Look up CUDA device IDs (though for MPS-only mode, we use simplified device IDs)

Even though MPS-only mode doesn't use MIG partitions, the file structure is still needed. We create a simplified version that satisfies the requirement.

```bash
# On Ubuntu
cd ~/GIT/socc22-miso

# Get your hostname
HOSTNAME=$(hostname)
echo "Your hostname is: $HOSTNAME"

# Create device mapping file for single GPU (adjust if you have multiple GPUs)
cat > mig_device_autogen.json << EOF
{
  "$HOSTNAME": {
    "gpu0": {
      "0": ["0"]
    }
  }
}
EOF

# Verify it was created correctly
cat mig_device_autogen.json
```

**For multiple GPUs**, modify the JSON:
```json
{
  "$HOSTNAME": {
    "gpu0": {
      "0": ["0"]
    },
    "gpu1": {
      "0": ["1"]
    }
  }
}
```

**Note**: The `enable_mps_simple.sh` script is already included in the repository. No need to create it manually.

**Note**: The `disable_mps_simple.sh` script is already included in the repository. No need to create it manually.

### 2.7 Verify gpu_server.py is MPS-Only Compatible

The `gpu_server.py` should already be modified for MPS-only mode. Verify these sections:

**Check line ~114-131 (mps_strt section):**
```python
elif 'mps_strt' in data_str:
    # Should have: device = str(gpuid)  # Direct GPU ID, no MIG lookup
    # Should NOT have: device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
```

**Check line ~132-150 (mps_rsm section):**
```python
elif 'mps_rsm' in data_str:
    # Should have: device = str(gpuid)  # Direct GPU ID, no MIG lookup
```

**Check line ~151-161 (mps_enable section):**
```python
elif 'mps_enable' in data_str:
    # Should have: cmd = f'./enable_mps_simple.sh {gpuid}'
    # Should NOT have: mig_helper.reset_mig() or mig_helper.create_ins()
```

If these are not correct, you may need to modify `gpu_server.py` manually (see `mps_quickstart.md` for exact changes).

### 2.8 Prepare Workloads

```bash
# On Ubuntu
cd ~/GIT/socc22-miso

# Create scratch directory (or use /tmp if /scratch doesn't exist)
sudo mkdir -p /scratch/$USER/miso_logs
sudo chown -R $USER:$USER /scratch/$USER

# Alternative: Use /tmp if /scratch doesn't exist
# mkdir -p /tmp/$USER/miso_logs
# ln -s /tmp/$USER/miso_logs /scratch/$USER/miso_logs  # Create symlink

mkdir -p /tmp/mps_log

# Download workload data (if needed)
# Install gdown if needed
pip install gdown

# Download workload data from Google Drive
gdown https://drive.google.com/uc?id=1pcPcPNdDRSYTMnwuibjBSeobm1tGFmxE

# Unzip
unzip MISO_Workload.zip

# Copy workloads to shared memory (if using shared memory)
# Update copy_memory.sh if needed
cat > workloads/copy_memory.sh << 'EOF'
#!/bin/bash
mkdir -p /dev/shm/tmp
cp -r ~/GIT/socc22-miso/MISO_Workload/ /dev/shm/tmp/MISO_Workload
EOF

chmod +x workloads/copy_memory.sh
chmod +x workloads/clear_memory.sh

# Copy workloads to shared memory
./workloads/clear_memory.sh
./workloads/copy_memory.sh
```

### 2.9 Configure Firewall

```bash
# On Ubuntu
# Allow TCP connections on port 10002
sudo ufw allow 10002/tcp

# Or if using iptables:
# sudo iptables -A INPUT -p tcp --dport 10002 -j ACCEPT

# Verify firewall status
sudo ufw status
```

### 2.10 Start GPU Server

**Option A: Using screen (Recommended)**

```bash
# On Ubuntu
cd ~/GIT/socc22-miso
conda activate tf2

# Start screen session
screen -S gpu_server

# Start GPU server
python gpu_server.py --node $(hostname) --port 10002

# Press Ctrl+A then D to detach from screen
# To reattach later: screen -r gpu_server
```

**Option B: Using tmux**

```bash
# On Ubuntu
cd ~/GIT/socc22-miso
conda activate tf2

# Start tmux session
tmux new -s gpu_server

# Start GPU server
python gpu_server.py --node $(hostname) --port 10002

# Press Ctrl+B then D to detach from tmux
# To reattach later: tmux attach -t gpu_server
```

**Option C: Using nohup (Background)**

```bash
# On Ubuntu
cd ~/GIT/socc22-miso
conda activate tf2

# Start GPU server in background
nohup python gpu_server.py --node $(hostname) --port 10002 > gpu_server.log 2>&1 &

# Check if it's running
ps aux | grep gpu_server

# View logs
tail -f gpu_server.log
```

**Verify GPU server is running:**

```bash
# On Ubuntu
# Check if process is running
ps aux | grep gpu_server

# Check if port is listening
netstat -tuln | grep 10002
# Or: ss -tuln | grep 10002
```

---

## Step 3: Run MPS Experiments

### 3.1 On macOS: Set GPU Server Hostname

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client  # or tf2

# Set GPU server hostname/IP as environment variable
export GPU_SERVER_HOST=<ubuntu-hostname-or-ip>
# Example: export GPU_SERVER_HOST=ubuntu@192.168.1.100
# Or: export GPU_SERVER_HOST=my-ubuntu-server

# Add to ~/.zshrc for persistence
echo 'export GPU_SERVER_HOST=<ubuntu-hostname-or-ip>' >> ~/.zshrc
```

### 3.2 Test Connection

```bash
# On macOS
# Test TCP connection to GPU server
nc -zv $GPU_SERVER_HOST 10002

# If connection succeeds, you're ready to run experiments!
```

### 3.3 Run Small Test Experiment

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client  # or tf2

# Run small test with 5 jobs
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 5 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST
```

### 3.4 Monitor Progress

**On macOS (Client):**
```bash
# Watch experiment log
tail -f logs/experiment_mps.log

# Check results directory
ls -la logs/mps/
```

**On Ubuntu (GPU Server):**
```bash
# SSH into Ubuntu server
ssh -i your-key.pem ubuntu@your-instance-ip

# Check GPU usage
watch -n 1 nvidia-smi

# Check running jobs
ps aux | grep python | grep train

# Check MPS status
nvidia-smi -q -d COMPUTE

# View GPU server logs (if using nohup)
tail -f ~/GIT/socc22-miso/gpu_server.log

# Or if using screen/tmux, attach to session and view output
```

### 3.5 Run Full Experiment

Once the test works, run the full experiment:

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client  # or tf2

# Run full experiment
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST
```

### 3.6 Run Experiments with Different MPS Levels

```bash
# On macOS
# Test different MPS thread percentages

# MPS Level 33 (33% threads)
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST

# MPS Level 50 (50% threads)
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 50 \
    --gpu_server_host $GPU_SERVER_HOST

# MPS Level 14 (14% threads)
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 14 \
    --gpu_server_host $GPU_SERVER_HOST
```

---

## Step 4: Results and Analysis

### 4.1 Results Location

After completion, find results on **macOS** (where scheduler runs):

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

# View results
ls -la logs/mps/

# Key result files:
# - logs/mps/JCT.json - Job Completion Times
# - logs/mps/JRT.json - Job Running Times  
# - logs/mps/QT.json - Queue Times
# - logs/mps/migration.json - Migration counts
# - logs/mps/active_jobs_per_gpu.json - Active jobs over time
# - logs/mps/completion.json - Completion status
# - logs/mps/progress.json - Progress tracking
# - logs/experiment_mps.log - Full experiment log
```

### 4.2 View Results

```bash
# On macOS
# View job completion times
cat logs/mps/JCT.json | python -m json.tool

# View experiment log
tail -100 logs/experiment_mps.log
```

---

## Troubleshooting

### macOS Client Issues

**Issue**: "Connection refused" on port 10002
- **Solution**: 
  - Verify GPU server is running on Ubuntu: `ps aux | grep gpu_server`
  - Check firewall on Ubuntu: `sudo ufw status`
  - Test connectivity: `nc -zv <ubuntu-ip> 10002`

**Issue**: "PackagesNotFoundError: python=3.7" on Apple Silicon (M1/M2/M3) Macs
- **Solution**: 
  - Python 3.7 is not available for ARM64. Use Python 3.8 or 3.9 instead:
    ```bash
    conda create -n miso_client python=3.8 -y
    # OR
    conda create -n miso_client python=3.9 -y
    ```
  - The scheduler code should work fine with Python 3.8 or 3.9

**Issue**: "Module not found" errors
- **Solution**: 
  - Verify conda environment is activated: `conda activate miso_client`
  - Install missing packages: `pip install numpy pandas pyyaml psutil`

**Issue**: "Cannot find device in cuda_devices"
- **Solution**: 
  - This error should not occur on macOS (it's a GPU server issue)
  - Verify you're running the scheduler, not GPU server code

**Issue**: Path-related errors
- **Solution**: 
  - The codebase uses `miso_config.py` for paths - no manual fixing needed
  - Verify `MISO_REPO_ROOT` is set (optional, auto-detects): `echo $MISO_REPO_ROOT`
  - Test path detection: `python -c "from miso_config import REPO_ROOT; print(REPO_ROOT)"`

### Ubuntu GPU Server Issues

**Issue**: "GPU must be in 7g.40gb to start MPS"
- **Solution**: The code should already be fixed for MPS-only mode. Verify `gpu_server.py` doesn't have MIG partition checks in `mps_strt` and `mps_rsm` sections.

**Issue**: "Cannot find device in cuda_devices"
- **Solution**: 
  - Verify `mig_device_autogen.json` has correct hostname: `cat mig_device_autogen.json`
  - Check hostname matches: `hostname` should match the key in JSON file

**Issue**: MPS not starting
- **Solution**:
  - Check sudo access: `sudo nvidia-smi`
  - Verify MPS directories exist: `ls -la /tmp/mps_log/`
  - Check MPS daemon: `ps aux | grep mps-control`
  - Manually test: `./enable_mps_simple.sh 0`

**Issue**: Jobs not starting
- **Solution**:
  - Check workload files exist: `ls -la ~/GIT/socc22-miso/MISO_Workload/`
  - Verify job models JSON: `cat mps/scheduler/simulator/job_models.json`
  - Check logs: `tail -f logs/experiment_mps.log` (on macOS) or `tail -f gpu_server.log` (on Ubuntu)

**Issue**: "Connection refused" on port 10002
- **Solution**:
  - Check GPU server is running: `ps aux | grep gpu_server`
  - Check firewall: `sudo ufw allow 10002/tcp`
  - Verify port: `netstat -tuln | grep 10002`

**Issue**: Permission denied for MPS scripts
- **Solution**:
  - Make scripts executable: `chmod +x enable_mps_simple.sh disable_mps_simple.sh`
  - Check sudo access: `sudo nvidia-smi`

**Issue**: `/scratch` directory doesn't exist
- **Solution**:
  - Create it: `sudo mkdir -p /scratch/$USER/miso_logs && sudo chown -R $USER:$USER /scratch/$USER`
  - Or modify `gpu_server.py` line 65 to use `/tmp` instead

### Network Issues

**Issue**: Cannot connect from macOS to Ubuntu
- **Solution**:
  - Verify Ubuntu IP/hostname is correct
  - Check if Ubuntu is behind NAT/firewall
  - Test basic connectivity: `ping <ubuntu-ip>`
  - Check if port 10002 is open: `nc -zv <ubuntu-ip> 10002`
  - Verify firewall rules on Ubuntu

**Issue**: Hostname resolution fails
- **Solution**:
  - Use IP address instead of hostname: `export GPU_SERVER_HOST=<ip-address>`
  - Or add to `/etc/hosts` on macOS: `sudo nano /etc/hosts` and add `<ubuntu-ip> <ubuntu-hostname>`

---

## Quick Reference Commands

### macOS Client

```bash
# Activate environment
conda activate miso_client

# Set GPU server hostname
export GPU_SERVER_HOST=<ubuntu-hostname-or-ip>

# Run experiment
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST

# Monitor logs
tail -f logs/experiment_mps.log
```

### Ubuntu GPU Server

```bash
# Activate environment
conda activate tf2

# Start GPU server (in screen/tmux)
screen -S gpu_server
python gpu_server.py --node $(hostname) --port 10002
# Ctrl+A then D to detach

# Monitor GPU
watch -n 1 nvidia-smi

# Check MPS status
nvidia-smi -q -d COMPUTE

# View server logs
tail -f gpu_server.log
```

---

## Summary Checklist

### macOS Client Setup
- [ ] Repository cloned
- [ ] Conda environment created (`miso_client`)
- [ ] Dependencies installed
- [ ] Path configuration verified (auto-detects, no fixing needed)
- [ ] `MISO_REPO_ROOT` environment variable set (optional)
- [ ] `run_mps_only.py` created
- [ ] `GPU_SERVER_HOST` environment variable set
- [ ] Network connectivity tested

### Ubuntu GPU Server Setup
- [ ] Repository cloned to `~/GIT/socc22-miso`
- [ ] Conda environment created (`tf2`)
- [ ] `MISO_REPO_ROOT` environment variable set
- [ ] `mig_device_autogen.json` created with correct hostname
- [ ] `enable_mps_simple.sh` created and executable
- [ ] `disable_mps_simple.sh` created and executable
- [ ] `gpu_server.py` verified for MPS-only mode
- [ ] Workload data downloaded and prepared
- [ ] Firewall configured (port 10002)
- [ ] GPU server started and running
- [ ] Port 10002 listening

### Running Experiments
- [ ] Small test (5 jobs) completed successfully
- [ ] Full experiment (30 jobs) completed
- [ ] Results saved in `logs/mps/` on macOS
- [ ] Different MPS levels tested (if desired)

---

## Next Steps

1. **Test with small job count first** (`--num_job 5`)
2. **Verify MPS is working**: Check `nvidia-smi -q -d COMPUTE` on Ubuntu
3. **Run full experiment** with desired parameters
4. **Compare results** across different MPS levels (33, 50, 14, etc.)
5. **Analyze results** in `logs/mps/` directory on macOS

---

## Additional Resources

- `environment_setup_guide.md` - Detailed environment setup comparison
- `macos_client_setup.md` - macOS client-specific setup
- `mps_only_plan.md` - Complete MPS-only setup plan
- `mps_quickstart.md` - Quick start guide for AWS L4
- `path_configuration_summary.md` - Path configuration details

---

**Happy Experimenting! 🚀**

