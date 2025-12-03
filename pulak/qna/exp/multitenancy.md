# MPS Level vs Max Tenants: Understanding the Difference

## Overview

In the MPS (Multi-Process Service) experiment framework, there are two key parameters that control different aspects of GPU resource sharing:

1. **`--mps_level`**: Controls **resource allocation per job** (thread percentage)
2. **`--max_tenants`**: Controls **number of concurrent jobs** per GPU

These parameters work together but control fundamentally different aspects of multitenancy.

---

## MPS Level (`--mps_level`)

### What It Controls
**MPS Level controls the percentage of GPU compute resources (SMs/threads) allocated to each individual job.**

### Technical Details
- Sets the `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable
- Determines what percentage of GPU streaming multiprocessors (SMs) each job can use
- Applied per job when it starts on the GPU

### How It Works
```python
# In workload files (e.g., resnet_train.py, transformer_train.py)
if args.mps_set: 
    os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = args.partition
```

When a job starts with `--partition 33` (or `--mps_level 33`), it sets:
```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33
```

### Example Values
- `--mps_level 33`: Each job gets 33% of GPU compute resources
- `--mps_level 50`: Each job gets 50% of GPU compute resources  
- `--mps_level 100`: Each job gets 100% of GPU compute resources

### Important Notes
- **MPS Level is per-job**: Each job running on the GPU gets this percentage
- **Total can exceed 100%**: If you have 3 jobs with `mps_level=50`, total allocation is 150% (they share and time-slice)
- **Affects performance**: Lower MPS level = less compute per job = slower execution per job
- **Memory is shared**: MPS level only controls compute (SMs), not memory

### Use Cases
- **Fine-grained resource control**: Allocate specific compute percentages to jobs
- **Performance tuning**: Balance between number of concurrent jobs and per-job performance
- **Fair sharing**: Ensure each job gets equal compute resources (e.g., 33% each for 3 jobs)

---

## Max Tenants (`--max_tenants`)

### What It Controls
**Max Tenants controls the maximum number of jobs that can run concurrently on the same GPU.**

### Technical Details
- Limits how many jobs the scheduler will pack onto a single GPU
- Controls the `full` property of `MPS_GPU_Status` class
- Determines when a GPU is considered "full" and cannot accept more jobs

### How It Works
```python
# In mps/scheduler/simulator/utils.py
class MPS_GPU_Status:    
    def __init__(self, node_index, max_tenants=3):
        self.max_tenants = max_tenants
    
    @property
    def full(self):
        if len(self.jobs) >= self.max_tenants:
            return True
        else:
            return False
```

### Example Values
- `--max_tenants 1`: Only 1 job can run on the GPU at a time (no sharing)
- `--max_tenants 2`: Up to 2 jobs can run concurrently
- `--max_tenants 3`: Up to 3 jobs can run concurrently (default)

### Important Notes
- **Max Tenants is per-GPU**: Controls how many jobs share a single GPU
- **Scheduling constraint**: Scheduler won't schedule more jobs than this limit
- **Independent of MPS level**: You can have 3 tenants with any MPS level
- **Affects concurrency**: Higher max_tenants = more jobs running simultaneously

### Use Cases
- **Multitenancy experiments**: Test different levels of job concurrency
- **Resource isolation**: Limit how many jobs share a GPU
- **Performance studies**: Compare 1 vs 2 vs 3 concurrent jobs

---

## Key Differences

| Aspect | MPS Level | Max Tenants |
|--------|-----------|-------------|
| **What it controls** | Compute resources per job | Number of concurrent jobs |
| **Unit** | Percentage (0-100) | Count (1, 2, 3, ...) |
| **Scope** | Per-job resource allocation | Per-GPU job limit |
| **Effect** | How much GPU each job gets | How many jobs share GPU |
| **Example** | `mps_level=33` → each job gets 33% | `max_tenants=2` → max 2 jobs |
| **Can exceed 100%?** | Yes (total across jobs) | No (hard limit) |

---

## How They Work Together

### Example Scenarios

#### Scenario 1: 2 Tenants, 50% MPS Level Each
```bash
--max_tenants 2 --mps_level 50
```
- **Result**: 2 jobs run concurrently, each gets 50% of GPU compute
- **Total allocation**: 100% (50% + 50%)
- **Behavior**: Jobs share GPU equally, no time-slicing needed

#### Scenario 2: 3 Tenants, 33% MPS Level Each
```bash
--max_tenants 3 --mps_level 33
```
- **Result**: 3 jobs run concurrently, each gets 33% of GPU compute
- **Total allocation**: 99% (33% + 33% + 33%)
- **Behavior**: Jobs share GPU with slight underutilization

#### Scenario 3: 3 Tenants, 50% MPS Level Each
```bash
--max_tenants 3 --mps_level 50
```
- **Result**: 3 jobs run concurrently, each gets 50% of GPU compute
- **Total allocation**: 150% (50% + 50% + 50%)
- **Behavior**: Jobs time-slice and compete for resources

#### Scenario 4: 1 Tenant, 100% MPS Level
```bash
--max_tenants 1 --mps_level 100
```
- **Result**: Only 1 job runs, gets 100% of GPU compute
- **Total allocation**: 100%
- **Behavior**: Exclusive GPU access, no sharing

---

## Practical Recommendations

### For Multitenancy Experiments
- **Keep MPS level constant** (e.g., `--mps_level 100`) and vary `--max_tenants` (1, 2, 3)
- This isolates the effect of concurrency on performance
- Example: `--mps_level 100 --max_tenants 1` vs `--max_tenants 2` vs `--max_tenants 3`

### For Resource Allocation Studies
- **Keep max_tenants constant** (e.g., `--max_tenants 3`) and vary `--mps_level` (33, 50, 100)
- This isolates the effect of per-job resource allocation
- Example: `--max_tenants 3 --mps_level 33` vs `--mps_level 50` vs `--mps_level 100`

### For Fair Sharing
- Use equal MPS levels: `--mps_level 33` with `--max_tenants 3` (33% each)
- Or: `--mps_level 50` with `--max_tenants 2` (50% each)

---

## Code References

### MPS Level Implementation
- **Set in**: `gpu_server.py` (line 121) - extracts from command
- **Passed to**: Workload scripts via `--partition` argument
- **Used in**: `workloads/*_train.py` - sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
- **Default**: 33 (in `run_mps_only.py`)

### Max Tenants Implementation
- **Defined in**: `run_mps_only.py` - `--max_tenants` argument
- **Passed to**: `exp_mps.py` - `MPS.__init__(max_tenants=...)`
- **Used in**: `mps/scheduler/simulator/utils.py` - `MPS_GPU_Status.full` property
- **Default**: 3 (hardcoded in original code, now configurable)

---

## Summary

- **MPS Level** = "How much GPU compute does each job get?" (percentage)
- **Max Tenants** = "How many jobs can run at the same time?" (count)

Both parameters are independent and can be combined to create different multitenancy scenarios. Understanding both is crucial for designing meaningful experiments.

