#!/bin/bash

# T4 Experiment Sweep: Resource Cap Sweep (Policy × MPS Level)
# Runs Phase T4 experiments with different MPS levels: {25, 33, 50, 100}
# Uses adversarial ordering (largest first → smallest last) from Phase T3.

set -e  # Exit on error

# Default parameters (can be overridden via command-line arguments)
ARRIVAL=${1:-100}
NUM_JOB=${2:-30}
SEED=${3:-42}
GPU_SERVER_HOST=${4:-localhost}
GPU_SERVER_PORT=${5:-10003}
NUM_GPU=${6:-1}
COLLECT_TELEMETRY=${7:-false}
TELEMETRY_INTERVAL=${8:-1.0}
MAX_TENANTS=${9:-3}

# MPS levels to sweep (as per experiment plan Phase T4)
MPS_LEVELS=(25 33 50 100)

# Results directory
RESULTS_BASE_DIR="logs/t4"
mkdir -p "$RESULTS_BASE_DIR"

echo "=========================================="
echo "T4 EXPERIMENT SWEEP: Resource Cap Sweep"
echo "=========================================="
echo "Parameters:"
echo "  - Arrival: $ARRIVAL"
echo "  - Jobs: $NUM_JOB"
echo "  - GPUs: $NUM_GPU"
echo "  - Seed: $SEED"
echo "  - Max Tenants: $MAX_TENANTS"
echo "  - Ordering: Adversarial (largest first → smallest last)"
echo "  - MPS Levels: ${MPS_LEVELS[*]}"
echo "  - GPU Server: $GPU_SERVER_HOST:$GPU_SERVER_PORT"
if [ "$COLLECT_TELEMETRY" = "true" ]; then
    echo "  - Telemetry: ENABLED (interval: ${TELEMETRY_INTERVAL}s)"
else
    echo "  - Telemetry: DISABLED"
fi
echo "  - Results Base: $RESULTS_BASE_DIR"
echo "=========================================="
echo ""

# Function to run a single experiment
run_experiment() {
    local mps_level=$1
    local output_dir="$RESULTS_BASE_DIR/mps_level_${mps_level}"
    
    echo "----------------------------------------"
    echo "Running: MPS Level = $mps_level"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build telemetry arguments
    TELEMETRY_ARGS=""
    if [ "$COLLECT_TELEMETRY" = "true" ]; then
        TELEMETRY_ARGS="--collect_telemetry --telemetry_interval $TELEMETRY_INTERVAL"
    fi
    
    # Run the experiment
    python3 run_mps_only.py \
        --arrival "$ARRIVAL" \
        --num_gpu "$NUM_GPU" \
        --num_job "$NUM_JOB" \
        --seed "$SEED" \
        --mps_level "$mps_level" \
        --max_tenants "$MAX_TENANTS" \
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
    
    echo "Completed MPS Level $mps_level"
    echo ""
}

# Run experiments for each MPS level
for mps_level in "${MPS_LEVELS[@]}"; do
    run_experiment "$mps_level"
    
    # Brief pause between experiments (except for the last one)
    if [ "$mps_level" != "${MPS_LEVELS[-1]}" ]; then
        echo "Waiting 10 seconds before next experiment..."
        sleep 10
        echo ""
    fi
done

echo "=========================================="
echo "All T4 experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
for mps_level in "${MPS_LEVELS[@]}"; do
    echo "  - $RESULTS_BASE_DIR/mps_level_${mps_level}/"
done
echo ""
echo "Each directory contains:"
echo "  - JCT.json, JRT.json, QT.json (performance metrics)"
echo "  - completion.json, progress.json (status tracking)"
echo "  - active_jobs_per_gpu.json (concurrency metrics)"
echo "  - migration.json, ckpt_dict.json, ckpt_ovhd.json (system metrics)"
echo "  - overall_rate.json (makespan)"
echo "  - experiment_mps.log (execution log)"
echo ""
echo "=========================================="
