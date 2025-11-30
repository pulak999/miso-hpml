# Running Three Workloads and Viewing Collected Data

This guide explains how to run exactly 3 workloads in an MPS experiment and view all the data being collected.

---

## How Workload Selection Works

### Overview

The MPS experiment system uses a **two-level mapping** to select workloads:

1. **Job Runtime Selection** (`trace_100.json`): Maps job indices (0-99) to runtime durations (in seconds)
2. **Workload Type Selection** (`job_models.json`): Maps job indices (0-99) to specific workload types

### Step-by-Step Process

#### 1. Job Creation (`exp_full.py` lines 35-47)

When you run with `--num_job 3`:

**Option A: Random Selection** (`--random_trace` flag):
```python
# Randomly samples 3 jobs from the 100 available
rand_ind = random.sample(range(len(job_dict)), args.num_job)  # e.g., [42, 7, 91]
for i in rand_ind:
    self.job_runtime[i] = int(job_dict[str(i)])  # Gets runtime from trace_100.json
```

**Option B: Sequential Selection** (default, no `--random_trace`):
```python
# Uses jobs 0, 1, 2 (cycles through 100 jobs)
for i in range(args.num_job):  # i = 0, 1, 2
    index = i % 100  # index = 0, 1, 2
    self.job_runtime[i] = int(job_dict[str(index)])  # Gets runtime from trace_100.json
```

#### 2. Workload Type Mapping (`exp_full.py` lines 105-108)

After jobs are created, each job gets mapped to a workload type:

```python
with open('mps/scheduler/simulator/job_models.json') as f:
    job_models = json.load(f)

# For job index i, lookup workload type:
mapped_jobid = str(int(jobid) % 100)  # Maps job ID to 0-99 range
workload_type = job_models[mapped_jobid]  # e.g., "transformer_train32"
```

**Example Mapping** (from `job_models.json`):
- Job 0 → `"transformer_train32"` (Transformer model, batch size 32)
- Job 1 → `"resnet_train128"` (ResNet model, batch size 128)
- Job 2 → `"transformer_train32"` (Transformer model, batch size 32)
- Job 3 → `"gnn_train512"` (GNN model, batch size 512)

#### 3. Workload Execution (`gpu_server.py` lines 114-131)

When a job starts, the GPU server:
1. Extracts the workload type from `job_models.json`
2. Parses model name and batch size
3. Executes the corresponding training script:

```python
mapped_jobid = str(int(jobid) % 100)
model = job_models[mapped_jobid].split('_')[0]  # e.g., "transformer"
batch = job_models[mapped_jobid].split('train')[1]  # e.g., "32"
iters = num_iters[mapped_jobid]  # Number of iterations from num_iters.json

cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} --node {host_node} --partition {mps_lvl} --mps_set --cuda_device {device}'
```

### Available Workload Types

From `job_models.json`, the system supports these workload types:

| Model | Batch Sizes Available |
|-------|----------------------|
| `transformer` | 16, 32, 64, 128 |
| `resnet` | 64, 128, 256, 512 |
| `gnn` | 128, 256, 512 |
| `embedding` | 64, 128, 256, 512 |
| `mobilenet` | 64, 128, 256, 512 |

**Total**: 100 unique workload configurations (indices 0-99)

---

## Running Exactly 3 Workloads

### Method 1: Use Sequential Jobs (Simplest)

This runs jobs 0, 1, 2 from the trace:

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client  # or tf2

# Run 3 jobs sequentially (jobs 0, 1, 2)
python run_mps_only.py \
    --arrival 30 \
    --num_gpu 1 \
    --num_job 3 \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST
```

**What this runs**:
- Job 0: `transformer_train32` (runtime: 120 seconds)
- Job 1: `resnet_train128` (runtime: 120 seconds)
- Job 2: `transformer_train32` (runtime: 120 seconds)

### Method 2: Use Random Selection

This randomly selects 3 jobs from the 100 available:

```bash
# On macOS
python run_mps_only.py \
    --arrival 30 \
    --num_gpu 1 \
    --num_job 3 \
    --random_trace \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST
```

**What this runs**: Random 3 jobs (e.g., jobs 42, 7, 91) with their corresponding workloads.

### Method 3: Select Specific Workloads (Advanced)

To run specific workloads, you need to modify the job selection logic. However, a simpler approach is to use `--random_trace` with a fixed seed to get reproducible random selection:

```bash
# Seed 42 might give you specific workloads - check logs to see which ones
python run_mps_only.py \
    --arrival 30 \
    --num_gpu 1 \
    --num_job 3 \
    --random_trace \
    --seed 12345 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST
```

**To see which workloads were selected**, check the experiment log:
```bash
tail -f logs/experiment_mps.log
# Look for lines like: "job 0 scheduled on GPU 0"
# Then check job_models.json to see which workload type job 0 maps to
```

---

## Data Collection Overview

The MPS experiment collects **10 types of metrics** during execution:

### 1. **JCT (Job Completion Time)**
- **Definition**: Time from job arrival to job completion
- **Formula**: `JCT[job] = comp_time[job] - arrive_time[job]`
- **File**: `logs/mps/JCT.json`
- **Units**: Seconds

### 2. **JRT (Job Running Time)**
- **Definition**: Time from job scheduling to job completion (actual execution time)
- **Formula**: `JRT[job] = comp_time[job] - sched_time[job]`
- **File**: `logs/mps/JRT.json`
- **Units**: Seconds

### 3. **QT (Queue Time)**
- **Definition**: Time from job arrival to job scheduling (waiting time)
- **Formula**: `QT[job] = sched_time[job] - arrive_time[job]`
- **File**: `logs/mps/QT.json`
- **Units**: Seconds

### 4. **Active Jobs Per GPU**
- **Definition**: Time series of average number of active jobs per GPU
- **File**: `logs/mps/active_jobs_per_gpu.json`
- **Format**: Array of values, one per simulation step
- **Units**: Average jobs per GPU (can be fractional)

### 5. **Completion Status**
- **Definition**: Per-job completion status (0 = not completed, 1 = completed)
- **File**: `logs/mps/completion.json`
- **Format**: `{"0": 1, "1": 1, "2": 1}` (1 = completed)

### 6. **Progress Tracking**
- **Definition**: Number of completed jobs at 60-second intervals
- **File**: `logs/mps/progress.json`
- **Format**: `{"60": 1, "120": 2, "180": 3}` (time: count)

### 7. **Migration Counts**
- **Definition**: Number of times each job was migrated (for MPS, usually 0)
- **File**: `logs/mps/migration.json`
- **Format**: `{"0": 0, "1": 0, "2": 0, "average": 0.0}`

### 8. **Checkpoint Dictionary**
- **Definition**: Number of checkpoints taken per job
- **File**: `logs/mps/ckpt_dict.json`
- **Format**: `{"0": 0, "1": 0, "2": 0}`

### 9. **Checkpoint Overhead**
- **Definition**: Time overhead for checkpoints per job
- **File**: `logs/mps/ckpt_ovhd.json`
- **Format**: `{"0": [], "1": [], "2": []}` (array of overhead times)

### 10. **Overall Rate**
- **Definition**: Overall processing rate (for MPS, this is the total experiment time)
- **File**: `logs/mps/overall_rate.json`
- **Format**: `[total_experiment_time_in_seconds]`

### Additional Logs

- **Experiment Log**: `logs/experiment_mps.log` - Detailed text log of all events
- **Job Output**: On GPU server: `/scratch/$USER/miso_logs/job{id}_start.out` and `.err`

---

## Viewing Collected Data

### Step 1: Wait for Experiment to Complete

The experiment will run until all 3 jobs complete. Monitor progress:

```bash
# On macOS - watch the experiment log
tail -f logs/experiment_mps.log

# Look for:
# - "job X scheduled on GPU Y"
# - "job X finished"
# - "all jobs are finished!"
```

### Step 2: View Results Files

After completion, all results are saved in `logs/mps/`:

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

# List all result files
ls -la logs/mps/

# Expected files:
# - JCT.json
# - JRT.json
# - QT.json
# - active_jobs_per_gpu.json
# - completion.json
# - progress.json
# - migration.json
# - ckpt_dict.json
# - ckpt_ovhd.json
# - overall_rate.json
```

### Step 3: View Individual Metrics

#### View Job Completion Times (JCT)

```bash
cat logs/mps/JCT.json | python -m json.tool
```

**Example Output**:
```json
{
    "0": 125,
    "1": 130,
    "2": 128,
    "average": 127.67
}
```

**Interpretation**:
- Job 0 took 125 seconds from arrival to completion
- Job 1 took 130 seconds
- Job 2 took 128 seconds
- Average: 127.67 seconds

#### View Job Running Times (JRT)

```bash
cat logs/mps/JRT.json | python -m json.tool
```

**Example Output**:
```json
{
    "0": 120,
    "1": 120,
    "2": 120,
    "average": 120.0
}
```

**Interpretation**:
- All jobs ran for exactly 120 seconds (matching trace_100.json)
- This is the actual execution time (excluding queue time)

#### View Queue Times (QT)

```bash
cat logs/mps/QT.json | python -m json.tool
```

**Example Output**:
```json
{
    "0": 0,
    "1": 5,
    "2": 8,
    "average": 4.33
}
```

**Interpretation**:
- Job 0 was scheduled immediately (no queue time)
- Job 1 waited 5 seconds before being scheduled
- Job 2 waited 8 seconds
- Average queue time: 4.33 seconds

#### View Active Jobs Per GPU

```bash
cat logs/mps/active_jobs_per_gpu.json | python -m json.tool | head -20
```

**Example Output**:
```json
[
    0.0,
    1.0,
    1.0,
    2.0,
    2.0,
    3.0,
    3.0,
    2.0,
    1.0,
    0.0
]
```

**Interpretation**:
- Each value is the average number of active jobs per GPU at that simulation step
- Values can be fractional if jobs are shared across GPUs
- Shows how jobs overlap in time

#### View Progress Over Time

```bash
cat logs/mps/progress.json | python -m json.tool
```

**Example Output**:
```json
{
    "60": 0,
    "120": 1,
    "180": 2,
    "240": 3
}
```

**Interpretation**:
- At 60 seconds: 0 jobs completed
- At 120 seconds: 1 job completed
- At 180 seconds: 2 jobs completed
- At 240 seconds: 3 jobs completed (all done)

#### View Completion Status

```bash
cat logs/mps/completion.json | python -m json.tool
```

**Example Output**:
```json
{
    "0": 1,
    "1": 1,
    "2": 1
}
```

**Interpretation**:
- All jobs (0, 1, 2) completed successfully (value = 1)

#### View Overall Experiment Time

```bash
cat logs/mps/overall_rate.json | python -m json.tool
```

**Example Output**:
```json
[
    245
]
```

**Interpretation**:
- Total experiment time: 245 seconds (from start to all jobs finished)

### Step 4: View Detailed Experiment Log

```bash
# View last 50 lines
tail -50 logs/experiment_mps.log

# View full log
cat logs/experiment_mps.log

# Search for specific job
grep "job 0" logs/experiment_mps.log
```

**Key Events to Look For**:
- `Schedule time: X` - When job was scheduled
- `job X scheduled on GPU Y` - Job assignment
- `job X finished` - Job completion
- `all jobs are finished!` - Experiment completion

### Step 5: View GPU Server Logs (Optional)

On the Ubuntu GPU server, you can also check individual job logs:

```bash
# SSH into GPU server
ssh -i your-key.pem ubuntu@your-instance-ip

# View job output logs
ls -la /scratch/$USER/miso_logs/

# View specific job's stdout
cat /scratch/$USER/miso_logs/job0_start.out

# View specific job's stderr
cat /scratch/$USER/miso_logs/job0_start.err
```

---

## Quick Example: Run and View 3 Workloads

### Complete Workflow

```bash
# 1. On macOS - Run experiment
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client
export GPU_SERVER_HOST=<your-ubuntu-host>

python run_mps_only.py \
    --arrival 30 \
    --num_gpu 1 \
    --num_job 3 \
    --seed 42 \
    --mps_level 33 \
    --gpu_server_host $GPU_SERVER_HOST

# 2. Wait for completion (watch log)
tail -f logs/experiment_mps.log
# Wait for "all jobs are finished!"

# 3. View all metrics
echo "=== Job Completion Times ==="
cat logs/mps/JCT.json | python -m json.tool

echo "=== Job Running Times ==="
cat logs/mps/JRT.json | python -m json.tool

echo "=== Queue Times ==="
cat logs/mps/QT.json | python -m json.tool

echo "=== Active Jobs Per GPU (first 10 values) ==="
cat logs/mps/active_jobs_per_gpu.json | python -m json.tool | head -15

echo "=== Progress Over Time ==="
cat logs/mps/progress.json | python -m json.tool

echo "=== Completion Status ==="
cat logs/mps/completion.json | python -m json.tool

echo "=== Overall Experiment Time ==="
cat logs/mps/overall_rate.json | python -m json.tool
```

---

## Understanding the Data

### Key Relationships

1. **JCT = QT + JRT**
   - Total time = Queue time + Running time
   - Example: JCT=125, QT=5, JRT=120 → 125 = 5 + 120 ✓

2. **JRT ≈ Runtime from trace_100.json**
   - Running time should match the runtime in the trace file
   - Small differences may occur due to MPS overhead

3. **Active Jobs Per GPU**
   - Shows concurrency: how many jobs run simultaneously
   - With MPS, multiple jobs can share a GPU
   - Values > 1 indicate job overlap

### What to Look For

**Good Signs**:
- ✅ All jobs complete (`completion.json` shows all 1s)
- ✅ JRT matches expected runtime from trace
- ✅ Active jobs per GPU shows proper sharing (values > 1 with MPS)
- ✅ No errors in experiment log

**Potential Issues**:
- ⚠️ High queue times (QT) - jobs waiting too long
- ⚠️ JRT much higher than trace runtime - possible performance degradation
- ⚠️ Completion status shows 0 - job didn't finish
- ⚠️ Errors in experiment log or GPU server logs

---

## Troubleshooting

### No Results Files Created

**Problem**: `logs/mps/` directory is empty after experiment

**Solutions**:
1. Check if experiment completed: `grep "all jobs are finished" logs/experiment_mps.log`
2. Check for errors: `grep -i error logs/experiment_mps.log`
3. Verify GPU server is running: `ps aux | grep gpu_server` (on Ubuntu)

### Jobs Not Completing

**Problem**: Jobs stuck, never finish

**Solutions**:
1. Check GPU server logs on Ubuntu: `tail -f /scratch/$USER/miso_logs/job0_start.err`
2. Verify workloads exist: `ls -la ~/GIT/socc22-miso/MISO_Workload/` (on Ubuntu)
3. Check MPS is enabled: `nvidia-smi -q -d COMPUTE` (on Ubuntu)

### Incomplete Data

**Problem**: Some metrics are missing or have 0 values

**Solutions**:
1. Check experiment log for errors
2. Verify all jobs completed: `cat logs/mps/completion.json`
3. Re-run experiment with `--num_job 3` to get fresh data

---

## Summary

**Workload Selection**:
- Jobs are selected from `trace_100.json` (runtime durations)
- Workload types are mapped from `job_models.json` (model + batch size)
- With `--num_job 3`, you get 3 jobs (indices 0, 1, 2 by default)

**Data Collected**:
- 10 JSON files with metrics (JCT, JRT, QT, etc.)
- 1 text log file with detailed events
- Job-specific logs on GPU server

**Viewing Data**:
- Use `cat logs/mps/<metric>.json | python -m json.tool` to view formatted JSON
- Check `logs/experiment_mps.log` for detailed event log
- Monitor GPU server logs for individual job output

---

**Happy Experimenting! 🚀**

