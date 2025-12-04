# Selecting 10 Specific Workloads

## Quick Guide

To run exactly 10 workloads of your choice, you have two main options:

---

## Option 1: Edit `job_models.json` (Recommended)

This is the simplest approach - edit the first 10 entries in `job_models.json`.

### Steps:

1. **Edit the file**: `mps/scheduler/simulator/job_models.json`

2. **Set indices 0-9 to your desired workloads**:
```json
{
  "0": "transformer_train32",
  "1": "resnet_train128",
  "2": "gnn_train512",
  "3": "embedding_train256",
  "4": "mobilenet_train64",
  "5": "transformer_train128",
  "6": "resnet_train256",
  "7": "gnn_train128",
  "8": "embedding_train512",
  "9": "mobilenet_train128",
  ...  // rest of the file stays the same
}
```

3. **Run your experiment**:
```bash
python3 run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 10 \
    --seed 42 \
    --mps_level 100 \
    --gpu_server_host localhost
```

**Without `--random_trace`**: Jobs 0-9 will use workloads from indices 0-9
**With `--random_trace`**: 10 random jobs will be selected, but they'll use your modified workloads

### Available Workload Types:

- `transformer_train{16,32,64,128}`
- `resnet_train{64,128,256,512}`
- `gnn_train{128,256,512}`
- `embedding_train{64,128,256,512}`
- `mobilenet_train{64,128,256,512}`

---

## Option 2: Use `--random_trace` with Fixed Seed

If you want to keep `job_models.json` unchanged but select specific workloads:

1. **Find which indices have your desired workloads** in `job_models.json`
   - Example: If you want `transformer_train32`, find all indices with that value

2. **Use `--random_trace --seed <value>` with `--num_job 10`**
   - This will randomly select 10 indices, but the same seed gives the same selection
   - You may need to try different seeds to get your desired combination

3. **Or modify the random selection logic** in `exp_full.py` (lines 39-43) to select specific indices

---

## Option 3: Create a Custom Workload Selection Script

Create a script that modifies `job_models.json` programmatically:

```python
import json
from pathlib import Path

# Your 10 chosen workloads
my_workloads = [
    "transformer_train32",
    "resnet_train128",
    "gnn_train512",
    "embedding_train256",
    "mobilenet_train64",
    "transformer_train128",
    "resnet_train256",
    "gnn_train128",
    "embedding_train512",
    "mobilenet_train128"
]

# Load existing job_models.json
job_models_path = Path("mps/scheduler/simulator/job_models.json")
with open(job_models_path) as f:
    job_models = json.load(f)

# Update first 10 entries
for i, workload in enumerate(my_workloads):
    job_models[str(i)] = workload

# Save back
with open(job_models_path, 'w') as f:
    json.dump(job_models, f, indent=4)

print("Updated job_models.json with your 10 workloads")
```

---

## Example: Running 10 Transformer Workloads

If you want to run 10 different transformer configurations:

1. **Edit `job_models.json`**:
```json
{
  "0": "transformer_train16",
  "1": "transformer_train32",
  "2": "transformer_train64",
  "3": "transformer_train128",
  "4": "transformer_train16",
  "5": "transformer_train32",
  "6": "transformer_train64",
  "7": "transformer_train128",
  "8": "transformer_train32",
  "9": "transformer_train64",
  ...  // rest unchanged
}
```

2. **Run**:
```bash
python3 run_mps_only.py \
    --num_job 10 \
    --num_gpu 1 \
    --arrival 100 \
    --mps_level 100 \
    --gpu_server_host localhost
```

---

## Verifying Your Selection

After running, check which workloads were actually used:

1. **Check experiment log**: `logs/experiment_mps.log`
   - Look for job scheduling messages

2. **Check GPU server logs**: On the server, check `/scratch/{user}/miso_logs/mps/`
   - Each job will have output files showing which model/batch was used

3. **Add logging** to see the mapping:
   - The GPU server prints which model/batch it's using when starting jobs

---

## Important Notes

1. **Workload Scripts Must Exist**: 
   - Each workload type needs a corresponding script: `{model}_train.py`
   - Located in `workloads/` directory
   - Example: `transformer_train32` → `workloads/transformer_train.py` with batch 32

2. **Batch Size Validation**:
   - Make sure the batch size in the workload name is valid for that model
   - Check `workloads/num_iters.json` to see which batch sizes are configured

3. **Backup Original File**:
   - Before editing, backup `job_models.json`:
   ```bash
   cp mps/scheduler/simulator/job_models.json mps/scheduler/simulator/job_models.json.backup
   ```

4. **Restore Original**:
   - After your experiment, restore if needed:
   ```bash
   cp mps/scheduler/simulator/job_models.json.backup mps/scheduler/simulator/job_models.json
   ```

---

## Quick Reference: Available Workloads

From `job_models.json`, here are all available workload types:

| Model | Batch Sizes Available |
|-------|----------------------|
| transformer | 16, 32, 64, 128 |
| resnet | 64, 128, 256, 512 |
| gnn | 128, 256, 512 |
| embedding | 64, 128, 256, 512 |
| mobilenet | 64, 128, 256, 512 |

**Format**: `{model}_train{batch_size}`

---

## Summary

**Simplest method**: Edit `mps/scheduler/simulator/job_models.json` and set indices 0-9 to your desired workloads, then run with `--num_job 10`.

