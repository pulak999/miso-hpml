# T4 Experiments: Resource Cap Sweep (Policy × MPS Level)

## Overview

Phase T4 experiments investigate the **fairness vs throughput trade-off** under different MPS resource caps. These experiments are part of the NVIDIA T4 scheduling policy sweeps described in the experiment plan.

## Experiment Design

**Objective:** Quantify the impact of resource capping (`mps_level`) on makespan, fairness, queueing behavior, and throughput under MPS-only sharing with adversarial job ordering.

### Key Parameters

- **Ordering:** Adversarial (largest first → smallest last) - same as Phase T3
- **MPS Level Sweep:** `{25, 33, 50, 100}`
  - `25`: 25% of GPU resources per job (4 jobs can share)
  - `33`: 33% of GPU resources per job (3 jobs can share)
  - `50`: 50% of GPU resources per job (2 jobs can share)
  - `100`: 100% of GPU resources per job (1 job at a time, effectively isolated)
- **Scheduling Policy:** Uses the best policy identified from Phase T2/T3
- **Focus:** Fairness vs throughput trade-off

### Expected Insights

1. **Lower MPS levels (25, 33):** Higher throughput but potential fairness issues
2. **Higher MPS levels (50, 100):** Better fairness but lower overall throughput
3. **Optimal trade-off:** Identify the MPS level that balances fairness and throughput

## Files

- `run_t4_sweep.py` - Python script to run all MPS level experiments
- `run_t4_sweep.sh` - Bash script alternative for running experiments
- `config_t4.json` - Configuration file with experiment parameters
- `README.md` - This file

## Usage

### Python Script (Recommended)

```bash
# Run all MPS level experiments
python t4/run_t4_sweep.py \
    --arrival 100 \
    --num_job 30 \
    --num_gpu 1 \
    --seed 42 \
    --gpu_server_host localhost \
    --gpu_server_port 10003

# With telemetry collection
python t4/run_t4_sweep.py \
    --arrival 100 \
    --num_job 30 \
    --num_gpu 1 \
    --seed 42 \
    --collect_telemetry \
    --telemetry_interval 1.0

# With W&B logging
python t4/run_t4_sweep.py \
    --arrival 100 \
    --num_job 30 \
    --num_gpu 1 \
    --seed 42 \
    --use_wandb \
    --wandb_project t4-mps-sweep \
    --wandb_tags t4 resource-cap adversarial
```

### Bash Script

```bash
# Make executable
chmod +x t4/run_t4_sweep.sh

# Run experiments
./t4/run_t4_sweep.sh [arrival] [num_job] [seed] [gpu_server_host] [gpu_server_port] [num_gpu] [collect_telemetry] [telemetry_interval]

# Example
./t4/run_t4_sweep.sh 100 30 42 localhost 10003 1 true 1.0
```

## Results

Results are saved in separate directories for each MPS level:

```
logs/t4/
├── mps_level_25/
│   ├── JCT.json
│   ├── JRT.json
│   ├── QT.json
│   ├── completion.json
│   ├── progress.json
│   ├── active_jobs_per_gpu.json
│   ├── migration.json
│   ├── ckpt_dict.json
│   ├── ckpt_ovhd.json
│   ├── overall_rate.json
│   └── experiment_mps.log
├── mps_level_33/
│   └── ...
├── mps_level_50/
│   └── ...
└── mps_level_100/
    └── ...
```

## Metrics Collected

- **JCT (Job Completion Time):** Time from arrival to completion
- **JRT (Job Running Time):** Time from scheduling to completion
- **QT (Queue Time):** Time from arrival to scheduling
- **Makespan:** Total time to complete all jobs
- **Fairness:** Measured via JCT variance and slowdown ratios
- **Throughput:** Jobs completed per unit time
- **Active Jobs per GPU:** Concurrency over time

## Analysis

After running experiments, compare:

1. **Makespan** across different MPS levels
2. **Fairness metrics** (JCT variance, max/min JCT ratio)
3. **Queue behavior** (average QT, max QT)
4. **Throughput** (jobs/second)
5. **Resource utilization** (active jobs per GPU over time)

## Notes

- These experiments use **adversarial ordering** (largest jobs first) to stress-test scheduling policies
- Ensure GPU server is running before starting experiments
- Each experiment may take 1-3 hours depending on workload size
- Results can be analyzed using the analysis scripts in the parent directory

## Related Experiments

- **Phase T0:** MPS isolated baseline
- **Phase T1:** Default (random arrival) baseline
- **Phase T2:** Scheduling policy sweep
- **Phase T3:** Adversarial ordering stress test (predecessor to T4)
