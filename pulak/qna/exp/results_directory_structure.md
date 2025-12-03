# Results Directory Structure

## Overview

When running `multitenancy_sweep.sh`, the script creates **6 separate directories**, one for each experiment configuration. Each directory contains all the results (JSON files and logs) for that specific experiment.

---

## Directory Structure

### Multitenancy Experiments (3 directories)

These experiments keep MPS level constant at 100 and vary the number of concurrent tenants:

1. **`logs/mps_multitenancy_1tenant/`**
   - Configuration: `--max_tenants 1 --mps_level 100`
   - Only 1 job runs at a time (no sharing)
   - Baseline for comparison

2. **`logs/mps_multitenancy_2tenants/`**
   - Configuration: `--max_tenants 2 --mps_level 100`
   - Up to 2 jobs run concurrently
   - Tests 2-tenant sharing

3. **`logs/mps_multitenancy_3tenants/`**
   - Configuration: `--max_tenants 3 --mps_level 100`
   - Up to 3 jobs run concurrently
   - Tests 3-tenant sharing

### Resource Allocation Experiments (3 directories)

These experiments keep max_tenants constant at 3 and vary the MPS level (resource allocation per job):

4. **`logs/mps_resource_alloc_mps33/`**
   - Configuration: `--max_tenants 3 --mps_level 33`
   - 3 jobs share GPU, each gets 33% compute
   - Fair sharing scenario

5. **`logs/mps_resource_alloc_mps50/`**
   - Configuration: `--max_tenants 3 --mps_level 50`
   - 3 jobs share GPU, each gets 50% compute
   - Over-subscription scenario (150% total)

6. **`logs/mps_resource_alloc_mps100/`**
   - Configuration: `--max_tenants 3 --mps_level 100`
   - 3 jobs share GPU, each gets 100% compute
   - Maximum over-subscription (300% total, heavy time-slicing)

---

## Contents of Each Directory

Each experiment directory contains the following files:

### Performance Metrics
- **`JCT.json`** - Job Completion Time (total time from arrival to completion)
- **`JRT.json`** - Job Running Time (actual execution time on GPU)
- **`QT.json`** - Queue Time (waiting time before scheduling)

### Status Tracking
- **`completion.json`** - Binary completion status for each job (1 = completed, 0 = failed)
- **`progress.json`** - Time-series of cumulative completion percentage

### Concurrency Metrics
- **`active_jobs_per_gpu.json`** - Time-series of average active jobs per GPU
  - Shows how many jobs were running concurrently over time
  - Values: 0.0 (idle), 1.0 (1 job), 2.0 (2 jobs), 3.0 (3 jobs)

### System Metrics
- **`migration.json`** - Number of job migrations (typically 0 for MPS-only)
- **`ckpt_dict.json`** - Checkpoint status (typically 0 for simple experiments)
- **`ckpt_ovhd.json`** - Checkpoint overhead times (typically empty)

### Overall Metrics
- **`overall_rate.json`** - Total experiment makespan (wall-clock time)

### Execution Log
- **`experiment_mps.log`** - Detailed execution log with scheduling events

---

## Aggregated Results Directory

All results are also copied to:
- **`logs/multitenancy_results/`**

This directory contains subdirectories:
- `multitenancy_1tenant/`
- `multitenancy_2tenants/`
- `multitenancy_3tenants/`
- `resource_alloc_mps33/`
- `resource_alloc_mps50/`
- `resource_alloc_mps100/`

This provides a centralized location for all experiment results.

---

## Important Notes

### Temporary Directory
The `logs/mps/` directory is **overwritten** by each experiment. It only contains the **last** experiment's results. Always use the experiment-specific directories (`logs/mps_*`) for analysis.

### Log File
The `logs/experiment_mps.log` file is also overwritten, but the script now copies it to each experiment directory, so you have a separate log for each experiment.

### Data Preservation
Each experiment's data is preserved in its own directory, so you can:
- Compare metrics across different configurations
- Analyze trends over time
- Re-run analysis without losing previous results

---

## Example: Comparing Results

To compare JCT across multitenancy levels:

```bash
# View JCT for 1 tenant
cat logs/mps_multitenancy_1tenant/JCT.json

# View JCT for 2 tenants
cat logs/mps_multitenancy_2tenants/JCT.json

# View JCT for 3 tenants
cat logs/mps_multitenancy_3tenants/JCT.json
```

To compare JRT across resource allocation levels:

```bash
# View JRT for MPS 33
cat logs/mps_resource_alloc_mps33/JRT.json

# View JRT for MPS 50
cat logs/mps_resource_alloc_mps50/JRT.json

# View JRT for MPS 100
cat logs/mps_resource_alloc_mps100/JRT.json
```

---

## Verification

After running the sweep script, verify all 6 directories exist:

```bash
ls -d logs/mps_*
```

You should see:
- `logs/mps_multitenancy_1tenant`
- `logs/mps_multitenancy_2tenants`
- `logs/mps_multitenancy_3tenants`
- `logs/mps_resource_alloc_mps33`
- `logs/mps_resource_alloc_mps50`
- `logs/mps_resource_alloc_mps100`

Each should contain all the JSON files and the log file.

