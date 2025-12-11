#!/usr/bin/env python3
"""
T4 Experiment Sweep: Resource Cap Sweep (Policy × MPS Level)

Runs Phase T4 experiments with different MPS levels: {25, 33, 50, 100}
Uses adversarial ordering (largest first → smallest last) from Phase T3.

This script runs multiple experiments, one for each MPS level, and saves
results in separate directories for easy comparison.
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
import json

# Add parent directory to path to import run_mps_only
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load unified config
try:
    from load_miso_config import get_config, get_gpu_server_config
    _config = get_config()
    _default_host, _default_port = get_gpu_server_config()
except (ImportError, Exception):
    _config = {}
    _default_host = None
    _default_port = 10003

# MPS levels to sweep (as per experiment plan Phase T4)
MPS_LEVELS = [25, 33, 50, 100]

def run_single_experiment(mps_level, args, results_base_dir):
    """
    Run a single experiment with a specific MPS level.
    
    Args:
        mps_level: MPS thread percentage (25, 33, 50, or 100)
        args: Parsed command-line arguments
        results_base_dir: Base directory for results
    
    Returns:
        Path to the results directory
    """
    # Create results directory for this MPS level
    results_dir = results_base_dir / f"mps_level_{mps_level}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Running T4 Experiment: MPS Level = {mps_level}")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  - Arrival: {args.arrival}")
    print(f"  - Jobs: {args.num_job}")
    print(f"  - GPUs: {args.num_gpu}")
    print(f"  - Seed: {args.seed}")
    print(f"  - MPS Level: {mps_level}")
    print(f"  - Max Tenants: {args.max_tenants}")
    print(f"  - Ordering: Adversarial (largest first → smallest last)")
    if args.collect_telemetry:
        print(f"  - Telemetry: ENABLED (interval: {args.telemetry_interval}s)")
    if args.use_wandb:
        print(f"  - W&B: ENABLED (project: {args.wandb_project})")
    print(f"  - Results: {results_dir}")
    print("=" * 80)
    print()
    
    # Build command arguments
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "run_mps_only.py"),
        "--arrival", str(args.arrival),
        "--num_job", str(args.num_job),
        "--num_gpu", str(args.num_gpu),
        "--seed", str(args.seed),
        "--step", str(args.step),
        "--mps_level", str(mps_level),
        "--max_tenants", str(args.max_tenants),
        "--gpu_server_host", args.gpu_server_host,
        "--gpu_server_port", str(args.gpu_server_port),
    ]
    
    # Add optional flags
    if args.random_trace:
        cmd.append("--random_trace")
    if args.filler:
        cmd.append("--filler")
    if args.flat_arrival:
        cmd.append("--flat_arrival")
    if args.collect_telemetry:
        cmd.append("--collect_telemetry")
        cmd.append("--telemetry_interval")
        cmd.append(str(args.telemetry_interval))
    if args.use_wandb:
        cmd.append("--use_wandb")
        cmd.append("--wandb_project")
        cmd.append(args.wandb_project)
        if args.wandb_entity:
            cmd.append("--wandb_entity")
            cmd.append(args.wandb_entity)
        if args.wandb_run_name:
            # Append MPS level to run name
            run_name = f"{args.wandb_run_name}_mps{mps_level}"
            cmd.append("--wandb_run_name")
            cmd.append(run_name)
        if args.wandb_tags:
            cmd.append("--wandb_tags")
            cmd.extend(args.wandb_tags)
            # Add MPS level tag
            cmd.append(f"mps{mps_level}")
    
    # Run the experiment
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        elapsed_time = time.time() - start_time
        
        # Copy results to experiment-specific directory
        source_logs = Path(__file__).parent.parent / "logs" / "mps"
        if source_logs.exists():
            print(f"\nCopying results to {results_dir}...")
            import shutil
            for file in source_logs.glob("*"):
                if file.is_file():
                    shutil.copy2(file, results_dir / file.name)
            
            # Also copy experiment log
            exp_log = Path(__file__).parent.parent / "logs" / "experiment_mps.log"
            if exp_log.exists():
                shutil.copy2(exp_log, results_dir / "experiment_mps.log")
        
        print(f"\n✓ Experiment completed in {elapsed_time:.1f} seconds")
        print(f"  Results saved to: {results_dir}")
        return results_dir
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Experiment failed after {elapsed_time:.1f} seconds")
        print(f"  Error code: {e.returncode}")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment interrupted by user")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='T4 Experiment Sweep: Resource Cap Sweep (MPS Level)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all MPS levels with default parameters
  python t4/run_t4_sweep.py --gpu_server_host localhost

  # Run with custom parameters
  python t4/run_t4_sweep.py \\
      --arrival 100 \\
      --num_job 30 \\
      --num_gpu 1 \\
      --seed 42 \\
      --gpu_server_host localhost \\
      --gpu_server_port 10003

  # Run with telemetry and W&B logging
  python t4/run_t4_sweep.py \\
      --arrival 100 \\
      --num_job 30 \\
      --collect_telemetry \\
      --use_wandb \\
      --wandb_project t4-mps-sweep \\
      --wandb_tags t4 resource-cap adversarial
        """
    )
    
    # Experiment parameters
    parser.add_argument('--arrival', type=int, default=_config.get('ARRIVAL', 100),
                        help='Inter-arrival period (default: 100)')
    parser.add_argument('--num_job', type=int, default=_config.get('NUM_JOB', 30),
                        help='Total number of jobs (default: 30)')
    parser.add_argument('--num_gpu', type=int, default=_config.get('NUM_GPU', 1),
                        help='Total number of GPUs (default: 1)')
    parser.add_argument('--seed', type=int, default=_config.get('SEED', 42),
                        help='Random seed (default: 42)')
    parser.add_argument('--step', type=int, default=10,
                        help='Simulation step size (default: 10)')
    parser.add_argument('--max_tenants', type=int, default=3,
                        help='Maximum concurrent tenants per GPU (default: 3)')
    
    # Job ordering (T4 uses adversarial ordering from T3)
    parser.add_argument('--random_trace', action='store_true', default=False,
                        help='Use random trace (default: False, uses adversarial ordering)')
    parser.add_argument('--filler', action='store_true', default=False,
                        help='Use filler jobs')
    parser.add_argument('--flat_arrival', action='store_true', default=False,
                        help='Use flat arrival pattern')
    
    # GPU server configuration
    parser.add_argument('--gpu_server_host', type=str, default=_default_host,
                        help='GPU server hostname or IP')
    parser.add_argument('--gpu_server_port', type=int, default=_default_port,
                        help='GPU server port')
    
    # Telemetry
    parser.add_argument('--collect_telemetry', action='store_true',
                        default=_config.get('COLLECT_TELEMETRY', False),
                        help='Enable GPU telemetry collection')
    parser.add_argument('--telemetry_interval', type=float,
                        default=_config.get('TELEMETRY_INTERVAL', 1.0),
                        help='Telemetry sampling interval in seconds (default: 1.0)')
    
    # W&B logging
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='t4-mps-sweep',
                        help='W&B project name (default: t4-mps-sweep)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name prefix (optional, MPS level will be appended)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                        help='W&B tags for all runs')
    
    # MPS level selection (optional, defaults to all)
    parser.add_argument('--mps_levels', type=int, nargs='+', default=MPS_LEVELS,
                        help=f'MPS levels to test (default: {MPS_LEVELS})')
    
    args = parser.parse_args()
    
    # Validate GPU server host
    if not args.gpu_server_host:
        print("ERROR: --gpu_server_host not specified")
        print("Options:")
        print("  1. Set in miso_config.sh (SSH_HOST, REMOTE_IP, USE_SSH_TUNNEL)")
        print("  2. Use --gpu_server_host <host> --gpu_server_port <port>")
        print("  3. Set GPU_SERVER_HOST environment variable")
        print("If using SSH tunnel, use: --gpu_server_host localhost --gpu_server_port 10003")
        sys.exit(1)
    
    # Validate MPS levels
    for level in args.mps_levels:
        if level not in MPS_LEVELS:
            print(f"WARNING: MPS level {level} not in standard set {MPS_LEVELS}")
            print("  Continuing anyway...")
    
    # Create results base directory
    results_base_dir = Path(__file__).parent.parent / "logs" / "t4"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Print experiment summary
    print("\n" + "=" * 80)
    print("T4 EXPERIMENT SWEEP: Resource Cap Sweep (Policy × MPS Level)")
    print("=" * 80)
    print(f"Experiment Parameters:")
    print(f"  - Arrival: {args.arrival}")
    print(f"  - Jobs: {args.num_job}")
    print(f"  - GPUs: {args.num_gpu}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Max Tenants: {args.max_tenants}")
    print(f"  - Ordering: {'Random' if args.random_trace else 'Adversarial (largest first)'}")
    print(f"  - MPS Levels: {args.mps_levels}")
    print(f"  - GPU Server: {args.gpu_server_host}:{args.gpu_server_port}")
    if args.collect_telemetry:
        print(f"  - Telemetry: ENABLED (interval: {args.telemetry_interval}s)")
    if args.use_wandb:
        print(f"  - W&B: ENABLED (project: {args.wandb_project})")
    print(f"  - Results Base: {results_base_dir}")
    print("=" * 80)
    print()
    
    # Run experiments for each MPS level
    results = {}
    start_time = time.time()
    
    for mps_level in sorted(args.mps_levels):
        result_dir = run_single_experiment(mps_level, args, results_base_dir)
        results[mps_level] = result_dir
        
        # Brief pause between experiments
        if mps_level != args.mps_levels[-1]:
            print("\n" + "-" * 80)
            print("Waiting 10 seconds before next experiment...")
            print("-" * 80 + "\n")
            time.sleep(10)
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXPERIMENT SWEEP COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nResults:")
    for mps_level, result_dir in sorted(results.items()):
        status = "✓" if result_dir else "✗"
        print(f"  {status} MPS Level {mps_level:3d}: {result_dir or 'FAILED'}")
    print(f"\nAll results saved in: {results_base_dir}")
    print("=" * 80)
    
    # Save experiment metadata
    metadata = {
        "experiment": "T4 Resource Cap Sweep",
        "parameters": {
            "arrival": args.arrival,
            "num_job": args.num_job,
            "num_gpu": args.num_gpu,
            "seed": args.seed,
            "max_tenants": args.max_tenants,
            "ordering": "adversarial" if not args.random_trace else "random",
            "mps_levels": args.mps_levels,
        },
        "results": {
            str(k): str(v) if v else None for k, v in results.items()
        },
        "total_time_seconds": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    metadata_file = results_base_dir / "experiment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nExperiment metadata saved to: {metadata_file}")

if __name__ == "__main__":
    main()
