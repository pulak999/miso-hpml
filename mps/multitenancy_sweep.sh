#!/bin/bash

# Multitenancy Sweep Script
# Runs two types of experiments:
# 1. Multitenancy Experiments: Vary max_tenants (1, 2, 3) with constant MPS level (100)
# 2. Resource Allocation Studies: Vary MPS level (33, 50, 100) with constant max_tenants (3)
# Each experiment saves results to a separate directory

set -e  # Exit on error

# Base parameters (you can modify these or pass them as arguments)
ARRIVAL=${1:-100}
NUM_JOB=${2:-10}
SEED=${3:-42}
GPU_SERVER_HOST=${4:-localhost}
GPU_SERVER_PORT=${5:-10003}
NUM_GPU=${6:-1}  # Using 1 GPU
COLLECT_TELEMETRY=${7:-true}  # Enable GPU telemetry by default (set to 'false' to disable)
TELEMETRY_INTERVAL=${8:-1.0}  # Telemetry sampling interval in seconds

# Results directory
RESULTS_DIR="logs/multitenancy_results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Multitenancy Sweep Experiment"
echo "=========================================="
echo "Jobs: $NUM_JOB"
echo "GPUs: $NUM_GPU"
echo "Arrival: $ARRIVAL"
echo "Seed: $SEED"
echo "GPU Telemetry: $COLLECT_TELEMETRY"
if [ "$COLLECT_TELEMETRY" = "true" ]; then
    echo "Telemetry Interval: ${TELEMETRY_INTERVAL}s"
fi
echo "GPU Server: $GPU_SERVER_HOST:$GPU_SERVER_PORT"
echo "=========================================="
echo ""

# Function to run experiment with specified parameters
run_experiment() {
    local experiment_name=$1
    local mps_level=$2
    local max_tenants=$3
    local output_dir="logs/mps_${experiment_name}"
    
    echo "----------------------------------------"
    echo "Running: $experiment_name"
    echo "MPS Level: $mps_level"
    echo "Max Tenants: $max_tenants"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build command with optional telemetry
    TELEMETRY_ARGS=""
    if [ "$COLLECT_TELEMETRY" = "true" ]; then
        TELEMETRY_ARGS="--collect_telemetry --telemetry_interval $TELEMETRY_INTERVAL"
    fi
    
    # Run the experiment
    python3 run_mps_only.py \
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
    
    # Copy results to experiment-specific directory
    if [ -d "logs/mps" ]; then
        echo "Copying results to $output_dir..."
        cp -r logs/mps/* "$output_dir/" 2>/dev/null || true
        echo "Results copied to $output_dir"
    else
        echo "Warning: logs/mps directory not found"
    fi
    
    # Also copy the experiment log file if it exists
    if [ -f "logs/experiment_mps.log" ]; then
        cp logs/experiment_mps.log "$output_dir/experiment_mps.log" 2>/dev/null || true
        echo "Experiment log copied to $output_dir/experiment_mps.log"
    fi
    
    echo "Completed $experiment_name"
    echo ""
}

# ============================================================
# Experiment Type 1: Multitenancy Experiments
# Keep MPS level constant (100), vary max_tenants (1, 2, 3)
# This isolates the effect of concurrency on performance
# ============================================================
echo "=========================================="
echo "EXPERIMENT TYPE 1: Multitenancy Studies"
echo "MPS Level: 100 (constant)"
echo "Max Tenants: 1, 2, 3 (varying)"
echo "=========================================="
echo ""

echo "Starting multitenancy experiment: 1 tenant..."
run_experiment "multitenancy_1tenant" 100 1

echo "Starting multitenancy experiment: 2 tenants..."
run_experiment "multitenancy_2tenants" 100 2

echo "Starting multitenancy experiment: 3 tenants..."
run_experiment "multitenancy_3tenants" 100 3

# ============================================================
# Experiment Type 2: Resource Allocation Studies
# Keep max_tenants constant (3), vary MPS level (33, 50, 100)
# This isolates the effect of per-job resource allocation
# ============================================================
echo "=========================================="
echo "EXPERIMENT TYPE 2: Resource Allocation Studies"
echo "Max Tenants: 3 (constant)"
echo "MPS Level: 33, 50, 100 (varying)"
echo "=========================================="
echo ""

echo "Starting resource allocation study: MPS level 33..."
run_experiment "resource_alloc_mps33" 33 3

echo "Starting resource allocation study: MPS level 50..."
run_experiment "resource_alloc_mps50" 50 3

echo "Starting resource allocation study: MPS level 100..."
run_experiment "resource_alloc_mps100" 100 3

# ============================================================
# Copy all results to final results directory
# ============================================================
echo "=========================================="
echo "Copying all results to $RESULTS_DIR"
echo "=========================================="

# Copy multitenancy experiments
for tenants in 1 2 3; do
    source_dir="logs/mps_multitenancy_${tenants}tenant"
    dest_dir="$RESULTS_DIR/multitenancy_${tenants}tenant"
    
    if [ -d "$source_dir" ]; then
        echo "Copying multitenancy ${tenants}-tenant results..."
        mkdir -p "$dest_dir"
        cp -r "$source_dir"/* "$dest_dir/" 2>/dev/null || true
        echo "  -> $dest_dir"
    fi
done

# Copy resource allocation experiments
for mps_level in 33 50 100; do
    source_dir="logs/mps_resource_alloc_mps${mps_level}"
    dest_dir="$RESULTS_DIR/resource_alloc_mps${mps_level}"
    
    if [ -d "$source_dir" ]; then
        echo "Copying resource allocation MPS${mps_level} results..."
        mkdir -p "$dest_dir"
        cp -r "$source_dir"/* "$dest_dir/" 2>/dev/null || true
        echo "  -> $dest_dir"
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Multitenancy Experiments (MPS=100, varying tenants):"
echo "  - logs/mps_multitenancy_1tenant/ (contains all JSON files + log)"
echo "  - logs/mps_multitenancy_2tenants/ (contains all JSON files + log)"
echo "  - logs/mps_multitenancy_3tenants/ (contains all JSON files + log)"
echo ""
echo "Resource Allocation Studies (tenants=3, varying MPS):"
echo "  - logs/mps_resource_alloc_mps33/ (contains all JSON files + log)"
echo "  - logs/mps_resource_alloc_mps50/ (contains all JSON files + log)"
echo "  - logs/mps_resource_alloc_mps100/ (contains all JSON files + log)"
echo ""
echo "Each directory contains:"
echo "  - JCT.json, JRT.json, QT.json (performance metrics)"
echo "  - completion.json, progress.json (status tracking)"
echo "  - active_jobs_per_gpu.json (concurrency metrics)"
echo "  - migration.json, ckpt_dict.json, ckpt_ovhd.json (system metrics)"
echo "  - overall_rate.json (makespan)"
echo "  - experiment_mps.log (execution log)"
echo ""
echo "All results also copied to: $RESULTS_DIR/"
echo "=========================================="
