#!/bin/bash
# ============================================================
# Unified MPS Configuration File
# ============================================================
# This is the master configuration file for the entire MPS setup.
# Set up once per server/client pair and all scripts will use it.
# ============================================================
# 
# Usage:
#   - Copy this file to miso_config.sh and customize it
#   - All scripts (setup_ssh_tunnels.sh, run_mps_only.py, etc.) will use it
#   - Or set MISO_CONFIG_PATH environment variable to point to your config
# ============================================================

# ============================================================
# Server Connection Configuration
# ============================================================
# SSH host alias (from ~/.ssh/config)
# Examples: "l4vm", "hpmlvm-hw1", "gpu-server", etc.
# Change this to switch between different servers
SSH_HOST="l4vm"

# Remote server IP address
# Get from: ssh $SSH_HOST "hostname -I" | awk '{print $1}'
REMOTE_IP="172.31.40.254"

# Use SSH tunnel? (true for private IPs, false for direct connection)
USE_SSH_TUNNEL=true

# ============================================================
# Port Configuration
# ============================================================
# GPU server port on remote machine
REMOTE_GPU_SERVER_PORT=10002

# Local port for forward tunnel (macOS -> Ubuntu)
# Use 10003 to avoid conflict with scheduler listener on 10002
FORWARD_TUNNEL_PORT=10003

# Local port for scheduler listener (macOS)
SCHEDULER_PORT=10002

# Reverse tunnel remote port (Ubuntu -> macOS)
REVERSE_TUNNEL_REMOTE_PORT=10002

# ============================================================
# Repository Configuration
# ============================================================
# Repository URL
REPO_URL="https://github.com/pulak999/miso-hpml.git"

# Client-side repository path (macOS)
CLIENT_REPO_ROOT="${MISO_REPO_ROOT:-~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso}"

# Server-side repository path (Ubuntu/GPU server)
SERVER_REPO_ROOT="${MISO_REPO_ROOT:-/home/ubuntu/GIT/socc22-miso}"

# Results directory on server (where experiments save results)
SERVER_RESULTS_BASE="/insomnia001/home/pm3371/mps_experiments"

# ============================================================
# Client Environment Configuration (macOS)
# ============================================================
CLIENT_CONDA_ENV="miso_client"  # or "tf2" if using full environment

CLIENT_CONDA_PATHS=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
)

# ============================================================
# Server Environment Configuration (Ubuntu/GPU Server)
# ============================================================
SERVER_CONDA_ENV="tf2"

# Module system (for clusters using environment modules)
# Set to empty string if not using modules
SERVER_MODULE_SYSTEM="anaconda"  # e.g., "anaconda", "python/3.9", or "" for no modules

SERVER_CONDA_PATHS=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
)

# ============================================================
# GPU Server Configuration
# ============================================================
# Number of GPUs
NUM_GPU=1

# GPU server hostname (use "localhost" for SLURM jobs on same node)
# For client scripts, this will be set to "localhost" when using SSH tunnel
GPU_SERVER_HOST="localhost"

# ============================================================
# Directory Configuration (Server)
# ============================================================
SCRATCH_BASE="/scratch"
TMP_BASE="/tmp"
MPS_LOG_DIR="/tmp/mps_log"

# ============================================================
# Experiment Default Parameters
# ============================================================
ARRIVAL=${ARRIVAL:-100}           # Inter-arrival period
NUM_JOB=${NUM_JOB:-30}             # Number of jobs
SEED=${SEED:-42}                    # Random seed
COLLECT_TELEMETRY=${COLLECT_TELEMETRY:-true}  # Enable GPU telemetry
TELEMETRY_INTERVAL=${TELEMETRY_INTERVAL:-1.0}  # Telemetry interval (seconds)

# ============================================================
# Server-Specific Settings
# ============================================================
GPU_SERVER_MAX_WAIT=30
UPDATE_REPO=true

# ============================================================
# Export all variables for use in scripts
# ============================================================
export SSH_HOST
export REMOTE_IP
export USE_SSH_TUNNEL
export REMOTE_GPU_SERVER_PORT
export FORWARD_TUNNEL_PORT
export SCHEDULER_PORT
export REVERSE_TUNNEL_REMOTE_PORT
export REPO_URL
export CLIENT_REPO_ROOT
export SERVER_REPO_ROOT
export SERVER_RESULTS_BASE
export CLIENT_CONDA_ENV
export SERVER_CONDA_ENV
export SERVER_MODULE_SYSTEM
export NUM_GPU
export GPU_SERVER_HOST
export SCRATCH_BASE
export TMP_BASE
export MPS_LOG_DIR
export ARRIVAL
export NUM_JOB
export SEED
export COLLECT_TELEMETRY
export TELEMETRY_INTERVAL
export GPU_SERVER_MAX_WAIT
export UPDATE_REPO

# Export arrays (convert to space-separated strings for compatibility)
export CLIENT_CONDA_PATHS="${CLIENT_CONDA_PATHS[*]}"
export SERVER_CONDA_PATHS="${SERVER_CONDA_PATHS[*]}"

