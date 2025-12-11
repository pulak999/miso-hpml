#!/bin/bash
# ============================================================
# Client Configuration File
# ============================================================
# This file contains client-specific configuration for MPS experiments.
# It loads from miso_config.sh first, then allows overrides.
# ============================================================

# ============================================================
# Load Unified Configuration (if available)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Try to load unified config first
if [ -n "$MISO_CONFIG_PATH" ] && [ -f "$MISO_CONFIG_PATH" ]; then
    source "$MISO_CONFIG_PATH"
elif [ -f "$REPO_ROOT_DIR/miso_config.sh" ]; then
    source "$REPO_ROOT_DIR/miso_config.sh"
elif [ -f "$HOME/.miso_config.sh" ]; then
    source "$HOME/.miso_config.sh"
fi

# ============================================================
# Client-Specific Overrides
# ============================================================
# Server Connection Configuration
SSH_HOST="${SSH_HOST:-l4vm}"
REMOTE_IP="${REMOTE_IP:-172.31.40.254}"
USE_SSH_TUNNEL="${USE_SSH_TUNNEL:-true}"

# Port Configuration
REMOTE_PORT="${REMOTE_PORT:-${REMOTE_GPU_SERVER_PORT:-10002}}"
FORWARD_TUNNEL_PORT="${FORWARD_TUNNEL_PORT:-10003}"
SCHEDULER_PORT="${SCHEDULER_PORT:-10002}"

# ============================================================
# Repository Configuration
# ============================================================
# Local repository root (on macOS/client)
REPO_ROOT="${REPO_ROOT:-${CLIENT_REPO_ROOT:-~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso}}"
REPO_URL="${REPO_URL:-https://github.com/pulak999/miso-hpml.git}"

# ============================================================
# Environment Configuration
# ============================================================
# Conda environment name (for client/scheduler)
CONDA_ENV="${CONDA_ENV:-${CLIENT_CONDA_ENV:-miso_client}}"

# Conda initialization paths
CONDA_PATHS=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
)

# ============================================================
# Experiment Default Parameters
# ============================================================
ARRIVAL=${ARRIVAL:-100}
NUM_JOB=${NUM_JOB:-30}
SEED=${SEED:-42}
COLLECT_TELEMETRY=${COLLECT_TELEMETRY:-true}
TELEMETRY_INTERVAL=${TELEMETRY_INTERVAL:-1.0}

# ============================================================
# Export variables
# ============================================================
export MISO_REPO_ROOT="$REPO_ROOT"
export SSH_HOST
export REMOTE_IP
export USE_SSH_TUNNEL
export REMOTE_PORT
export FORWARD_TUNNEL_PORT
export SCHEDULER_PORT
export REPO_URL
export CONDA_ENV
export ARRIVAL
export NUM_JOB
export SEED
export COLLECT_TELEMETRY
export TELEMETRY_INTERVAL

