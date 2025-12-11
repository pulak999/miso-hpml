#!/bin/bash
#
#SBATCH --account=free       # The account name for the job.
#SBATCH --job-name=mps-experiments   # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 8                     # The number of cpu cores to use.
#SBATCH --time=0-12:00           # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=10gb        # The memory the job will use per cpu core.

set -e  # Exit on error (but allow function returns)

# ============================================================
# Load Configuration
# ============================================================
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load server configuration
# First try to load from script directory, then from current directory
if [ -f "$SCRIPT_DIR/server_config.sh" ]; then
    source "$SCRIPT_DIR/server_config.sh"
    echo "Loaded configuration from: $SCRIPT_DIR/server_config.sh"
elif [ -f "./insomnia/server_config.sh" ]; then
    source "./insomnia/server_config.sh"
    echo "Loaded configuration from: ./insomnia/server_config.sh"
elif [ -f "$HOME/.miso_server_config.sh" ]; then
    source "$HOME/.miso_server_config.sh"
    echo "Loaded configuration from: $HOME/.miso_server_config.sh"
else
    echo "WARNING: server_config.sh not found. Using defaults."
    echo "Please create server_config.sh in the insomnia/ directory or ~/.miso_server_config.sh"
    # Set minimal defaults
    REPO_ROOT="${MISO_REPO_ROOT:-/insomnia001/home/pm3371/socc22-miso}"
    RESULTS_BASE="/insomnia001/home/pm3371/mps_experiments"
    GPU_SERVER_PORT=10002
    GPU_SERVER_HOST="localhost"
    NUM_GPU=1
    CONDA_ENV="tf2"
    MODULE_SYSTEM="anaconda"
    ARRIVAL=${ARRIVAL:-100}
    NUM_JOB=${NUM_JOB:-10}
    SEED=${SEED:-9}
    COLLECT_TELEMETRY=${COLLECT_TELEMETRY:-true}
    TELEMETRY_INTERVAL=${TELEMETRY_INTERVAL:-60.0}
    REPO_URL="https://github.com/pulak999/miso-hpml.git"
    UPDATE_REPO=true
    GPU_SERVER_MAX_WAIT=30
fi

echo "=========================================="
echo "MPS Experiments SLURM Job"
echo "=========================================="
echo "Repository: $REPO_ROOT"
echo "Results: $RESULTS_BASE"
echo "GPU Server: $GPU_SERVER_HOST:$GPU_SERVER_PORT"
echo "Jobs: $NUM_JOB, Arrival: $ARRIVAL, Seed: $SEED"
echo "=========================================="
echo ""

# ============================================================
# Step 0: Clone/Update Repository
# ============================================================
echo "Step 0: Setting up repository..."

# Load module if specified
if [ -n "$MODULE_SYSTEM" ]; then
    echo "Loading module: $MODULE_SYSTEM..."
    module load "$MODULE_SYSTEM" || echo "WARNING: Failed to load module $MODULE_SYSTEM"
fi
REPO_DIR=$(dirname "$REPO_ROOT")
REPO_NAME=$(basename "$REPO_ROOT")

# Create parent directory if it doesn't exist
mkdir -p "$REPO_DIR"

# Clone or update repository
if [ -d "$REPO_ROOT" ]; then
    echo "Repository already exists at $REPO_ROOT"
    if [ "$UPDATE_REPO" = "true" ]; then
        echo "Updating repository..."
        cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }
        
        # Check if it's a git repository
        if [ -d ".git" ]; then
            # Update existing repository
            git fetch origin || echo "WARNING: git fetch failed, continuing anyway"
            git reset --hard origin/main 2>/dev/null || git reset --hard origin/master 2>/dev/null || echo "WARNING: git reset failed, continuing anyway"
            echo "Repository updated"
        else
            echo "WARNING: $REPO_ROOT exists but is not a git repository"
            echo "Skipping update, using existing directory"
        fi
    else
        echo "UPDATE_REPO=false, skipping repository update"
    fi
else
    echo "Cloning repository from $REPO_URL to $REPO_ROOT..."
    cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR"; exit 1; }
    git clone "$REPO_URL" "$REPO_NAME" || {
        echo "ERROR: Failed to clone repository"
        exit 1
    }
    echo "Repository cloned successfully"
fi

# ============================================================
# Step 1: Setup Environment
# ============================================================
echo ""
echo "Step 1: Setting up environment..."

# Change to repository directory
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Verify we're in the right directory
if [ ! -f "run_mps_only.py" ] || [ ! -f "gpu_server.py" ]; then
    echo "ERROR: Required files not found in $REPO_ROOT"
    echo "Expected files: run_mps_only.py, gpu_server.py"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Set MISO_REPO_ROOT if not already set
export MISO_REPO_ROOT="$REPO_ROOT"

# Initialize conda (check paths from config or use defaults)
CONDA_INITIALIZED=false
if [ -n "${CONDA_PATHS[*]}" ]; then
    for conda_path in "${CONDA_PATHS[@]}"; do
        if [ -f "$conda_path" ]; then
            source "$conda_path"
            CONDA_INITIALIZED=true
            echo "Initialized conda from: $conda_path"
            break
        fi
    done
else
    # Fallback to default paths
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        CONDA_INITIALIZED=true
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        CONDA_INITIALIZED=true
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
        CONDA_INITIALIZED=true
    fi
fi

if [ "$CONDA_INITIALIZED" = "false" ]; then
    echo "WARNING: Could not find conda initialization script"
    echo "Trying to use conda from PATH..."
fi

# Activate conda environment
if command -v conda &> /dev/null; then
    if conda env list | grep -q "^${CONDA_ENV} "; then
        conda activate "$CONDA_ENV"
        echo "Activated conda environment: $CONDA_ENV"
    else
        echo "ERROR: Conda environment '$CONDA_ENV' not found"
        echo "Available environments:"
        conda env list
        exit 1
    fi
else
    echo "ERROR: conda command not found"
    exit 1
fi

# Verify GPU access
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================
# Step 2: Create Device Mapping Configuration
# ============================================================
echo ""
echo "Step 2: Creating device mapping configuration..."

HOSTNAME=$(hostname)
echo "Hostname: $HOSTNAME"

# Create mig_device_autogen.json if it doesn't exist or update it
if [ ! -f "mig_device_autogen.json" ] || ! grep -q "\"$HOSTNAME\"" mig_device_autogen.json 2>/dev/null; then
    echo "Creating/updating mig_device_autogen.json..."
    cat > mig_device_autogen.json << EOF
{
  "$HOSTNAME": {
    "gpu0": {
      "0": ["0"]
    }
  }
}
EOF
    echo "Created mig_device_autogen.json with hostname: $HOSTNAME"
else
    echo "mig_device_autogen.json already exists with correct hostname"
fi

# ============================================================
# Step 3: Prepare Workloads and Directories
# ============================================================
echo ""
echo "Step 3: Preparing workloads and directories..."

# Create necessary directories
mkdir -p logs/mps
mkdir -p "$MPS_LOG_DIR"
mkdir -p "$RESULTS_BASE"

# Create scratch directory if needed (for gpu_server.py logs)
USER=$(whoami)
if [ ! -d "$SCRATCH_BASE/$USER/miso_logs" ]; then
    mkdir -p "$SCRATCH_BASE/$USER/miso_logs" 2>/dev/null || mkdir -p "$TMP_BASE/$USER/miso_logs"
    if [ -d "$TMP_BASE/$USER/miso_logs" ]; then
        echo "Using $TMP_BASE/$USER/miso_logs for logs"
    fi
fi

# ============================================================
# Step 4: Start GPU Server
# ============================================================
echo ""
echo "Step 4: Starting GPU server..."

# Kill any existing gpu_server processes on this port
lsof -ti:$GPU_SERVER_PORT | xargs kill -9 2>/dev/null || true
sleep 2

# Start GPU server in background
echo "Starting gpu_server.py on $GPU_SERVER_HOST:$GPU_SERVER_PORT..."
nohup python gpu_server.py --node "$HOSTNAME" --port "$GPU_SERVER_PORT" > gpu_server.log 2>&1 &
GPU_SERVER_PID=$!
echo "GPU server started with PID: $GPU_SERVER_PID"

# Wait for GPU server to be ready
echo "Waiting for GPU server to be ready..."
MAX_WAIT="${GPU_SERVER_MAX_WAIT:-30}"
WAIT_COUNT=0

# Function to check if port is open
check_port() {
    local host=$1
    local port=$2
    # Try nc first, then python as fallback
    if command -v nc &> /dev/null; then
        nc -z "$host" "$port" 2>/dev/null
    else
        python -c "import socket; s = socket.socket(); s.settimeout(1); result = s.connect_ex(('$host', $port)); s.close(); exit(0 if result == 0 else 1)" 2>/dev/null
    fi
}

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if check_port "$GPU_SERVER_HOST" "$GPU_SERVER_PORT"; then
        echo "GPU server is ready!"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    echo -n "."
done
echo ""

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: GPU server did not start within $MAX_WAIT seconds"
    echo "GPU server log:"
    tail -50 gpu_server.log
    exit 1
fi

# ============================================================
# Step 5: Run Experiments
# ============================================================
echo ""
echo "=========================================="
echo "Step 5: Running Experiments"
echo "=========================================="

# Function to run experiment and save results
run_experiment() {
    local experiment_name=$1
    local mps_level=$2
    local max_tenants=$3
    local output_dir="$RESULTS_BASE/${experiment_name}"
    
    echo ""
    echo "----------------------------------------"
    echo "Running: $experiment_name"
    echo "MPS Level: $mps_level"
    echo "Max Tenants: $max_tenants"
    echo "Output: $output_dir"
    echo "----------------------------------------"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build command with optional telemetry
    TELEMETRY_ARGS=""
    if [ "$COLLECT_TELEMETRY" = "true" ]; then
        TELEMETRY_ARGS="--collect_telemetry --telemetry_interval $TELEMETRY_INTERVAL"
    fi
    
    # Run the experiment (disable set -e for this command to handle errors gracefully)
    set +e
    python run_mps_only.py \
        --arrival "$ARRIVAL" \
        --num_gpu "$NUM_GPU" \
        --num_job "$NUM_JOB" \
        --random_trace \
        --seed "$SEED" \
        --mps_level "$mps_level" \
        --max_tenants "$max_tenants" \
        $TELEMETRY_ARGS \
        --gpu_server_host "$GPU_SERVER_HOST" \
        --gpu_server_port "$GPU_SERVER_PORT"
    EXP_RESULT=$?
    set -e
    
    if [ $EXP_RESULT -ne 0 ]; then
        echo "ERROR: Experiment $experiment_name failed with exit code $EXP_RESULT"
        echo "Continuing with next experiment..."
        return 1
    fi
    
    # Copy results to persistent storage
    if [ -d "logs/mps" ]; then
        echo "Copying results to $output_dir..."
        cp -r logs/mps/* "$output_dir/" 2>/dev/null || true
        echo "Results copied to $output_dir"
    else
        echo "Warning: logs/mps directory not found"
    fi
    
    # Copy experiment log
    if [ -f "logs/experiment_mps.log" ]; then
        cp logs/experiment_mps.log "$output_dir/experiment_mps.log" 2>/dev/null || true
        echo "Experiment log copied to $output_dir/experiment_mps.log"
    fi
    
    # Copy GPU server log for this experiment
    if [ -f "gpu_server.log" ]; then
        cp gpu_server.log "$output_dir/gpu_server_${experiment_name}.log" 2>/dev/null || true
    fi
    
    echo "Completed $experiment_name"
}

# ============================================================
# Experiment 1: Multitenancy Studies
# Vary max_tenants (1, 2, 3) with constant MPS level (100)
# ============================================================
echo ""
echo "=========================================="
echo "EXPERIMENT 1: Multitenancy Studies"
echo "MPS Level: 100 (constant)"
echo "Max Tenants: 1, 2, 3 (varying)"
echo "=========================================="

run_experiment "experiment1_multitenancy_1tenant" 100 1
run_experiment "experiment1_multitenancy_2tenants" 100 2
run_experiment "experiment1_multitenancy_3tenants" 100 3

# ============================================================
# Experiment 2: Resource Allocation Studies
# Vary MPS level (33, 50, 100) with constant max_tenants (3)
# ============================================================
echo ""
echo "=========================================="
echo "EXPERIMENT 2: Resource Allocation Studies"
echo "Max Tenants: 3 (constant)"
echo "MPS Level: 33, 50, 100 (varying)"
echo "=========================================="

run_experiment "experiment2_resource_alloc_mps33" 33 3
run_experiment "experiment2_resource_alloc_mps50" 50 3
run_experiment "experiment2_resource_alloc_mps100" 100 3

# ============================================================
# Step 6: Final Results Summary
# ============================================================
echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_BASE"
echo ""
echo "Experiment 1 - Multitenancy (MPS=100, varying tenants):"
echo "  - $RESULTS_BASE/experiment1_multitenancy_1tenant/"
echo "  - $RESULTS_BASE/experiment1_multitenancy_2tenants/"
echo "  - $RESULTS_BASE/experiment1_multitenancy_3tenants/"
echo ""
echo "Experiment 2 - Resource Allocation (tenants=3, varying MPS):"
echo "  - $RESULTS_BASE/experiment2_resource_alloc_mps33/"
echo "  - $RESULTS_BASE/experiment2_resource_alloc_mps50/"
echo "  - $RESULTS_BASE/experiment2_resource_alloc_mps100/"
echo ""
echo "Each directory contains:"
echo "  - JCT.json, JRT.json, QT.json (performance metrics)"
echo "  - completion.json, progress.json (status tracking)"
echo "  - active_jobs_per_gpu.json (concurrency metrics)"
echo "  - migration.json, ckpt_dict.json, ckpt_ovhd.json (system metrics)"
echo "  - overall_rate.json (makespan)"
echo "  - experiment_mps.log (execution log)"
echo "  - gpu_server_*.log (GPU server log)"
echo ""

# ============================================================
# Step 7: Cleanup
# ============================================================
echo "Step 7: Cleaning up..."

# Stop GPU server
if kill -0 "$GPU_SERVER_PID" 2>/dev/null; then
    echo "Stopping GPU server (PID: $GPU_SERVER_PID)..."
    kill "$GPU_SERVER_PID" 2>/dev/null || true
    sleep 2
    # Force kill if still running
    if kill -0 "$GPU_SERVER_PID" 2>/dev/null; then
        kill -9 "$GPU_SERVER_PID" 2>/dev/null || true
    fi
fi

# Kill any remaining processes on the port
lsof -ti:$GPU_SERVER_PORT | xargs kill -9 2>/dev/null || true

echo "Cleanup completed"
echo ""
echo "=========================================="
echo "Job finished successfully!"
echo "==========================================" 