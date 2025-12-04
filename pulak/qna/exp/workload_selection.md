# Workload Selection in MPS Experiments

## Overview

Workloads are selected through a **two-step mapping process**:

1. **Job Creation**: Jobs are created from a trace file
2. **Workload Mapping**: Each job ID maps to a workload type via `job_models.json`

---

## Step 1: Job Creation

### Source: `trace_100.json`

**Location**: `mps/scheduler/trace/trace_100.json`

This file contains **job runtime information** for 100 different job configurations. Each entry specifies how long a job should run (in some time unit).

**Example**:
```json
{
  "0": 150,
  "1": 200,
  "2": 180,
  ...
  "99": 175
}
```

### How Jobs Are Created

In `exp_full.py` (lines 38-47):

**With `--random_trace` flag** (when `num_job <= 100`):
```python
# Randomly sample from trace_100.json
rand_ind = random.sample(range(len(job_dict)), args.num_job)
for i in rand_ind:
    self.job_runtime[i] = int(job_dict[str(i)])
```
- Randomly selects `num_job` entries from the trace
- Job IDs are the selected indices (e.g., if indices [5, 23, 42] are selected, jobs 5, 23, 42 are created)

**Without `--random_trace` or when `num_job > 100`**:
```python
# Cycle through trace_100.json
for i in range(args.num_job):
    index = i % 100
    self.job_runtime[i] = int(job_dict[str(index)])
```
- Creates jobs 0, 1, 2, ..., num_job-1
- Maps each job ID to trace index using modulo: `job_id % 100`
- Job 0 → trace[0], Job 1 → trace[1], ..., Job 100 → trace[0] (wraps around)

---

## Step 2: Workload Type Mapping

### Source: `job_models.json`

**Location**: `mps/scheduler/simulator/job_models.json`

This file maps job indices (0-99) to workload types. Each workload type specifies:
- **Model name**: e.g., `transformer`, `resnet`, `gnn`, `embedding`, `mobilenet`
- **Batch size**: e.g., `32`, `64`, `128`, `256`, `512`

**Format**: `{model}_train{batch_size}`

**Example**:
```json
{
  "0": "transformer_train32",
  "1": "resnet_train128",
  "2": "transformer_train32",
  "3": "gnn_train512",
  ...
  "99": "embedding_train256"
}
```

### How Workload Is Selected

When a job is scheduled on the GPU server (`gpu_server.py` lines 125-130):

```python
# Map job ID to workload index (0-99)
mapped_jobid = str(int(jobid) % 100)

# Look up workload type
workload_type = job_models[mapped_jobid]  # e.g., "transformer_train32"

# Parse workload type
model = workload_type.split('_')[0]      # e.g., "transformer"
batch = workload_type.split('train')[1]  # e.g., "32"
```

**Key Point**: The mapping uses `jobid % 100`, so:
- Job 0 → workload index 0 → `job_models["0"]`
- Job 1 → workload index 1 → `job_models["1"]`
- Job 100 → workload index 0 → `job_models["0"]` (same as job 0)
- Job 101 → workload index 1 → `job_models["1"]` (same as job 1)

---

## Complete Flow Example

### Example: Job 81

1. **Job Creation**:
   - Job 81 is created
   - Runtime: `trace_100.json["81"]` = some runtime value

2. **Workload Mapping**:
   - `mapped_jobid = 81 % 100 = 81`
   - `workload_type = job_models["81"]` = `"resnet_train64"`

3. **Workload Execution**:
   - Model: `resnet`
   - Batch: `64`
   - Script: `resnet_train.py`
   - Command: `python resnet_train.py --job_id 81 -b 64 ...`

---

## Available Workload Types

From `job_models.json`, the system supports:

- **transformer**: Transformer model training
- **resnet**: ResNet model training
- **gnn**: Graph Neural Network training
- **embedding**: Embedding model training
- **mobilenet**: MobileNet model training

**Batch sizes**: 16, 32, 64, 128, 256, 512

**Total configurations**: 100 unique workload configurations (indices 0-99)

---

## Modifying Workload Selection

### To Change Which Workloads Run

**Option 1: Modify `job_models.json`**
- Edit the mapping for specific indices
- Example: Change `"81": "resnet_train64"` to `"81": "transformer_train128"`

**Option 2: Use `--random_trace` with Fixed Seed**
- Use `--random_trace --seed <value>` to get reproducible random selection
- The same seed will select the same jobs (and thus workloads)

**Option 3: Modify Job Creation Logic**
- Edit `exp_full.py` to change how jobs are created
- You could filter or modify which jobs are selected

---

## Important Notes

1. **Job ID vs Workload Index**: 
   - Job IDs can be any number (0, 1, 2, ..., num_job-1)
   - Workload indices are always 0-99 (via `jobid % 100`)

2. **Workload Reuse**:
   - Jobs with IDs that differ by multiples of 100 use the same workload
   - Job 0, 100, 200 all use `job_models["0"]`

3. **Random Selection**:
   - With `--random_trace`, jobs are randomly selected from indices 0-99
   - This means you might get duplicate workloads if the same index is selected multiple times

4. **Workload Scripts**:
   - Each workload type must have a corresponding script: `{model}_train.py`
   - Scripts are located in `workloads/` directory
   - Example: `transformer_train32` → `workloads/transformer_train.py` with batch size 32

---

## Summary

**Workloads are selected from**:
1. **`trace_100.json`**: Provides job runtime information (100 entries)
2. **`job_models.json`**: Maps job indices (0-99) to workload types (100 entries)

**Selection process**:
1. Jobs are created (either randomly or sequentially)
2. Each job ID maps to workload index: `jobid % 100`
3. Workload index maps to workload type: `job_models[str(workload_index)]`
4. Workload type is parsed to get model and batch size
5. Corresponding `{model}_train.py` script is executed

**To see which workloads will run**, check:
- `mps/scheduler/simulator/job_models.json` - All available workload mappings
- Experiment logs - Which jobs were actually scheduled

