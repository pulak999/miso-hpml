# T4 Experiments Quick Start Guide

## Prerequisites

1. **GPU Server Running**: Ensure `gpu_server.py` is running on your GPU machine
2. **SSH Tunnel** (if using remote GPU): Set up reverse SSH tunnel:
   ```bash
   ssh -R 10003:localhost:10003 <gpu-server>
   ```
3. **Environment**: Activate your conda environment:
   ```bash
   conda activate tf2  # or your environment name
   ```

## Quick Run

### Option 1: Python Script (Recommended)

```bash
cd /path/to/socc22-miso

# Basic run with default parameters
python t4/run_t4_sweep.py --gpu_server_host localhost

# Custom parameters
python t4/run_t4_sweep.py \
    --arrival 100 \
    --num_job 30 \
    --num_gpu 1 \
    --seed 42 \
    --gpu_server_host localhost \
    --gpu_server_port 10003
```

### Option 2: Bash Script

```bash
cd /path/to/socc22-miso

# Basic run
./t4/run_t4_sweep.sh

# Custom parameters
./t4/run_t4_sweep.sh 100 30 42 localhost 10003 1 false 1.0 3
```

## What Gets Run

The script automatically runs 4 experiments, one for each MPS level:
- **MPS Level 25**: 25% resources per job
- **MPS Level 33**: 33% resources per job  
- **MPS Level 50**: 50% resources per job
- **MPS Level 100**: 100% resources per job (isolated)

Each experiment uses **adversarial ordering** (largest jobs first).

## Results Location

Results are saved in:
```
logs/t4/
├── mps_level_25/
├── mps_level_33/
├── mps_level_50/
└── mps_level_100/
```

Each directory contains:
- `JCT.json` - Job Completion Times
- `JRT.json` - Job Running Times
- `QT.json` - Queue Times
- `completion.json` - Completion status
- `progress.json` - Progress over time
- `active_jobs_per_gpu.json` - Concurrency metrics
- `experiment_mps.log` - Full execution log

## With Telemetry

To collect GPU telemetry (DCGM + NVML):

```bash
python t4/run_t4_sweep.py \
    --gpu_server_host localhost \
    --collect_telemetry \
    --telemetry_interval 1.0
```

## With W&B Logging

To log experiments to Weights & Biases:

```bash
python t4/run_t4_sweep.py \
    --gpu_server_host localhost \
    --use_wandb \
    --wandb_project t4-mps-sweep \
    --wandb_tags t4 resource-cap adversarial
```

## Running Specific MPS Levels

To run only specific MPS levels:

```bash
python t4/run_t4_sweep.py \
    --gpu_server_host localhost \
    --mps_levels 33 50 100
```

## Expected Runtime

- Each experiment: ~1-3 hours (depends on `num_job` and workload)
- Full sweep (4 experiments): ~4-12 hours
- Monitor progress: `tail -f logs/experiment_mps.log`

## Troubleshooting

### "GPU server host not specified"
- Set `--gpu_server_host localhost` (or your server IP)
- Or set in `miso_config.sh`

### "Connection refused"
- Ensure GPU server is running: `ps aux | grep gpu_server`
- Check SSH tunnel if using remote GPU

### "No jobs starting"
- Check GPU server logs
- Verify job listener is running
- Check `logs/experiment_mps.log` for errors

## Next Steps

After experiments complete:
1. Compare makespan across MPS levels
2. Analyze fairness metrics (JCT variance)
3. Plot throughput vs fairness trade-off
4. Identify optimal MPS level for your workload

See `README.md` for detailed analysis guidelines.
