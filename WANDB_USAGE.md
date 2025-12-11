# Weights & Biases (wandb) Integration

This setup now supports logging experiment metrics to Weights & Biases (wandb) for tracking and visualization.

## Installation

Install wandb in your environment:

```bash
pip install wandb
```

Or if using conda:

```bash
conda install -c conda-forge wandb
```

## Usage

### Basic Usage

Enable wandb logging by adding the `--use_wandb` flag:

```bash
python run_mps_only.py \
    --gpu_server_host localhost \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 5 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --use_wandb
```

### Advanced Usage

Customize wandb project, run name, and tags:

```bash
python run_mps_only.py \
    --gpu_server_host localhost \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --use_wandb \
    --wandb_project my-mps-experiments \
    --wandb_run_name test_run_001 \
    --wandb_tags mps33 small_test baseline
```

### Command-Line Arguments

- `--use_wandb`: Enable wandb logging (default: False)
- `--wandb_project`: W&B project name (default: "mps-experiments")
- `--wandb_entity`: W&B entity/team name (optional)
- `--wandb_run_name`: Custom run name (optional, auto-generated if not provided)
- `--wandb_tags`: Space-separated list of tags for this run (optional)

## What Gets Logged

### Configuration
- MPS level, number of jobs, GPUs, arrival rate, seed, etc.
- All experiment parameters are logged as wandb config

### Time-Series Metrics (logged during experiment)
- `gpu/active_jobs_per_gpu`: Average active jobs per GPU over time
- `gpu/total_active_jobs`: Total active jobs across all GPUs
- `progress/completed_jobs`: Number of completed jobs
- `progress/completion_rate`: Completion percentage
- `time/elapsed_seconds`: Elapsed time since experiment start

### Final Metrics (logged at experiment end)
- `metrics/jct_mean`, `metrics/jct_std`: Job Completion Time statistics
- `metrics/jrt_mean`, `metrics/jrt_std`: Job Running Time statistics
- `metrics/qt_mean`, `metrics/qt_std`: Queue Time statistics
- `metrics/migration_mean`, `metrics/migration_total`: Migration statistics
- `metrics/makespan_seconds`: Total experiment duration
- `metrics/total_jobs`, `metrics/completed_jobs`: Job counts

### Tables
- `active_jobs_per_gpu_table`: Time-series of active jobs per GPU
- `progress_table`: Progress over time
- `per_job_metrics`: Per-job JCT, JRT, QT, and migration counts

### Histograms
- `histograms/jct`: Distribution of Job Completion Times
- `histograms/jrt`: Distribution of Job Running Times
- `histograms/qt`: Distribution of Queue Times

## Viewing Results

1. **Web Dashboard**: Results are automatically synced to your wandb dashboard at https://wandb.ai
2. **Local Mode**: Use `wandb offline` to run without syncing, then `wandb sync` later
3. **Compare Runs**: Use wandb's compare feature to analyze different experiments

## Example: Comparing MPS Levels

```bash
# Run with MPS level 33
python run_mps_only.py --use_wandb --wandb_run_name mps33 --mps_level 33 --wandb_tags mps33 ...

# Run with MPS level 50
python run_mps_only.py --use_wandb --wandb_run_name mps50 --mps_level 50 --wandb_tags mps50 ...

# Compare in wandb dashboard
```

## Notes

- Wandb logging is **optional** - existing JSON file logging continues to work
- If wandb is not installed, the experiment will continue without wandb logging (with a warning)
- All metrics are still saved to `logs/mps/*.json` files regardless of wandb status
- Wandb automatically handles experiment tracking, visualization, and comparison

