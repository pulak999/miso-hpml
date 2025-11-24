# MPS-Only Quick Start Guide - AWS L4 (1 GPU)

This is a simplified quick-start guide for running MPS experiments on a single AWS L4 GPU instance.

## Quick Answers

1. **GPUs**: 1 L4 GPU → Use `--num_gpu 1`
2. **Workload Data**: Download from Google Drive (link below)
3. **Hostname**: Run `hostname` command on your instance
4. **MPS Level**: Start with 33 (simplest)

## Step-by-Step Setup

### 1. SSH and Verify

```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Check GPU
nvidia-smi

# Find hostname (you'll need this)
hostname
# Save this output - you'll use it later
```

### 2. Setup Repository

```bash
mkdir -p ~/GIT
cd ~/GIT

# Clone or upload repository
# If you have it locally:
scp -r -i your-key.pem /local/path/to/socc22-miso ubuntu@your-instance-ip:~/GIT/

cd ~/GIT/socc22-miso

# Create conda environment
conda env create -f environment.yml
conda activate <env-name>  # Check environment.yml for exact name
```

### 3. Download Workload Data

```bash
cd ~/GIT/socc22-miso

# Install gdown if needed
pip install gdown

# Download workload data from Google Drive
gdown https://drive.google.com/uc?id=1pcPcPNdDRSYTMnwuibjBSeobm1tGFmxE

# If gdown doesn't work, download manually from browser and upload:
# scp -i your-key.pem MISO_Workload.zip ubuntu@your-instance-ip:~/GIT/socc22-miso/

# Unzip
unzip MISO_Workload.zip
```

### 4. Create Configuration Files

#### 4.1 Device Mapping (1 GPU)

```bash
cd ~/GIT/socc22-miso

# Get your hostname
HOSTNAME=$(hostname)
echo "Using hostname: $HOSTNAME"

# Create device mapping
cat > mig_device_autogen.json << EOF
{
  "$HOSTNAME": {
    "gpu0": {
      "0": ["0"]
    }
  }
}
EOF
```

#### 4.2 MPS Enable Script

```bash
cat > enable_mps_simple.sh << 'EOF'
#!/bin/bash
GPU_ID=$1
sudo nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS
mkdir -p /tmp/mps_log/nvidia-mps$GPU_ID
mkdir -p /tmp/mps_log/nvidia-log$GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_log/nvidia-mps$GPU_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log/nvidia-log$GPU_ID
nvidia-cuda-mps-control -d
echo "MPS enabled for GPU $GPU_ID"
EOF

chmod +x enable_mps_simple.sh
```

#### 4.3 MPS Disable Script

```bash
cat > disable_mps_simple.sh << 'EOF'
#!/bin/bash
GPU_ID=$1
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i $GPU_ID -c DEFAULT
echo "MPS disabled for GPU $GPU_ID"
EOF

chmod +x disable_mps_simple.sh
```

#### 4.4 Update Workload Scripts

```bash
# Update copy_memory.sh
cat > workloads/copy_memory.sh << 'EOF'
#!/bin/bash
mkdir -p /dev/shm/tmp
cp -r ~/GIT/socc22-miso/MISO_Workload/ /dev/shm/tmp/MISO_Workload
EOF

chmod +x workloads/copy_memory.sh
chmod +x workloads/clear_memory.sh

# Copy workloads to shared memory
./workloads/clear_memory.sh
./workloads/copy_memory.sh
```

### 5. Modify gpu_server.py

You need to modify `gpu_server.py` to remove MIG dependencies. The key changes are:

**Around line 141-150** (mps_enable section):

**REPLACE THIS:**
```python
elif 'mps_enable' in data_str: # mps_enable 0
    gpuid = int(re.findall(r'\d+', data_str)[0])
    mig_helper.reset_mig(gpuid)
    mig_helper.create_ins(gpuid, '7g.40gb')
    current_partition[gpuid] = '0'
    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
    cmd = f'./enable_mps_on_mig.sh {device}'
    p = subprocess.Popen([cmd], shell=True)
    p.wait()
    print(f'enabled MPS on GPU {gpuid}')
```

**WITH THIS:**
```python
elif 'mps_enable' in data_str: # mps_enable 0
    gpuid = int(re.findall(r'\d+', data_str)[0])
    # REMOVED: mig_helper.reset_mig(gpuid)
    # REMOVED: mig_helper.create_ins(gpuid, '7g.40gb')
    current_partition[gpuid] = '0'
    # REMOVED: device lookup from cuda_devices
    cmd = f'./enable_mps_simple.sh {gpuid}'
    p = subprocess.Popen([cmd], shell=True, cwd=f'/home/{user}/GIT/socc22-miso')
    p.wait()
    print(f'enabled MPS on GPU {gpuid}')
```

**Around line 102-120** (mps_strt section):

**REPLACE THIS:**
```python
elif 'mps_strt' in data_str: # mps_strt 10 gpu 0 lvl 100
    jobid = re.findall(r'\d+', data_str)[0]
    gpuid = int(re.findall(r'\d+', data_str)[1])
    mps_lvl = re.findall(r'\d+', data_str)[2]
    if current_partition[gpuid] != '0':
        raise RuntimeError('GPU must be in 7g.40gb to start MPS')
    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
    mapped_jobid = str(int(jobid) % 100)
    model = job_models[mapped_jobid].split('_')[0]
    batch = job_models[mapped_jobid].split('train')[1]
    iters = num_iters[mapped_jobid]
    cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} \
        --node {host_node} --partition {mps_lvl} --mps_set --cuda_device {device}'
    print(f'starting job {jobid} at gpu {gpuid} for MPS')
    
    out_file = f'{log_dir}/job{jobid}_start.out'
    err_file = f'{log_dir}/job{jobid}_start.err'
    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
```

**WITH THIS:**
```python
elif 'mps_strt' in data_str: # mps_strt 10 gpu 0 lvl 100
    jobid = re.findall(r'\d+', data_str)[0]
    gpuid = int(re.findall(r'\d+', data_str)[1])
    mps_lvl = re.findall(r'\d+', data_str)[2]
    # REMOVED: MIG partition check (lines 106-107)
    device = str(gpuid)  # Direct GPU ID, no MIG lookup
    mapped_jobid = str(int(jobid) % 100)
    model = job_models[mapped_jobid].split('_')[0]
    batch = job_models[mapped_jobid].split('train')[1]
    iters = num_iters[mapped_jobid]
    cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} \
        --node {host_node} --partition {mps_lvl} --mps_set --cuda_device {device}'
    print(f'starting job {jobid} at gpu {gpuid} for MPS')
    
    out_file = f'{log_dir}/job{jobid}_start.out'
    err_file = f'{log_dir}/job{jobid}_start.err'
    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
```

**Around line 121-140** (mps_rsm section):

**REPLACE THIS:**
```python
elif 'mps_rsm' in data_str: # mps_rsm 10 gpu 0 batch 500 lvl 100
    jobid = re.findall(r'\d+', data_str)[0]
    gpuid = int(re.findall(r'\d+', data_str)[1])
    resume_batch = int(re.findall(r'\d+', data_str)[2])
    mps_lvl = re.findall(r'\d+', data_str)[3]
    if current_partition[gpuid] != '0':
        raise RuntimeError('GPU must be in 7g.40gb to start MPS')
    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
    mapped_jobid = str(int(jobid) % 100)
    model = job_models[mapped_jobid].split('_')[0]
    batch = job_models[mapped_jobid].split('train')[1]
    iters = num_iters[mapped_jobid]
    cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} \
        --node {host_node} --partition {mps_lvl} --mps_set --resume --start_batch {resume_batch} --cuda_device {device}'
    print(f'resuming job {jobid} at gpu {gpuid} for MPS')

    out_file = f'{log_dir}/job{jobid}_resume.out'
    err_file = f'{log_dir}/job{jobid}_resume.err'     
    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
```

**WITH THIS:**
```python
elif 'mps_rsm' in data_str: # mps_rsm 10 gpu 0 batch 500 lvl 100
    jobid = re.findall(r'\d+', data_str)[0]
    gpuid = int(re.findall(r'\d+', data_str)[1])
    resume_batch = int(re.findall(r'\d+', data_str)[2])
    mps_lvl = re.findall(r'\d+', data_str)[3]
    # REMOVED: MIG partition check
    device = str(gpuid)  # Direct GPU ID, no MIG lookup
    mapped_jobid = str(int(jobid) % 100)
    model = job_models[mapped_jobid].split('_')[0]
    batch = job_models[mapped_jobid].split('train')[1]
    iters = num_iters[mapped_jobid]
    cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} \
        --node {host_node} --partition {mps_lvl} --mps_set --resume --start_batch {resume_batch} --cuda_device {device}'
    print(f'resuming job {jobid} at gpu {gpuid} for MPS')

    out_file = f'{log_dir}/job{jobid}_resume.out'
    err_file = f'{log_dir}/job{jobid}_resume.err'     
    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
```

**Also remove MIG helper import** (around line 15):
```python
# Remove or comment out:
# import mig_helper
```

### 6. Create Run Script

```bash
cat > run_mps_only.py << 'EOF'
#!/usr/bin/env python3
import argparse
import time
import socket
from pathlib import Path
from exp_mps import MPS

parser = argparse.ArgumentParser(description='MPS-only experiment')
parser.add_argument('--arrival', type=int, default=60, help='inter-arrival period')
parser.add_argument('--num_job', type=int, default=30, help='total number of jobs')
parser.add_argument('--num_gpu', type=int, default=1, help='total number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--step', type=int, default=10, help='simulation step size')
parser.add_argument('--filler', action='store_true', default=False)
parser.add_argument('--flat_arrival', action='store_true', default=False)
parser.add_argument('--random_trace', action='store_true', default=False)
parser.add_argument('--mps_level', type=int, default=33, help='MPS thread percentage')
args = parser.parse_args()

Path('logs/mps').mkdir(parents=True, exist_ok=True)

# Get hostname dynamically
hostname = socket.gethostname()
physical_nodes = [hostname]
print(f'Using hostname: {hostname}')

print('=' * 60)
print('Running MPS-Only Experiment')
print(f'Jobs: {args.num_job}, GPUs: {args.num_gpu}, MPS Level: {args.mps_level}')
print('=' * 60)

mps_exp = MPS(args, physical_nodes)
mps_exp.run(args, mps_lvl=args.mps_level)

print('Experiment completed!')
print(f'Results saved in: logs/mps/')
EOF

chmod +x run_mps_only.py
```

### 7. Start GPU Server

In a **separate terminal** or use `screen`/`tmux`:

```bash
# Option 1: Using screen
screen -S gpu_server
cd ~/GIT/socc22-miso
conda activate <env-name>
python gpu_server.py --node $(hostname) --port 10002
# Press Ctrl+A then D to detach

# Option 2: Using tmux
tmux new -s gpu_server
cd ~/GIT/socc22-miso
conda activate <env-name>
python gpu_server.py --node $(hostname) --port 10002
# Press Ctrl+B then D to detach

# Option 3: Background process
cd ~/GIT/socc22-miso
conda activate <env-name>
nohup python gpu_server.py --node $(hostname) --port 10002 > gpu_server.log 2>&1 &
```

### 8. Run Experiment

```bash
cd ~/GIT/socc22-miso
conda activate <env-name>

# Test with small number of jobs first
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 5 \
    --random_trace \
    --seed 42 \
    --mps_level 33

# If test works, run full experiment
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 33
```

### 9. Monitor Progress

```bash
# Watch experiment log
tail -f logs/experiment_mps.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check MPS status
nvidia-smi -q -d COMPUTE

# Check running jobs
ps aux | grep python | grep train
```

## Troubleshooting

### "GPU must be in 7g.40gb to start MPS"
**Fix**: Remove the MIG partition check in `gpu_server.py` lines 106-107 and 126-127.

### "Cannot find device in cuda_devices"
**Fix**: Verify `mig_device_autogen.json` has correct hostname. Run `hostname` and update the file.

### "Connection refused" on port 10002
**Fix**: 
- Check GPU server is running: `ps aux | grep gpu_server`
- Check port: `netstat -tuln | grep 10002`
- Restart GPU server if needed

### Workload files not found
**Fix**: 
- Verify MISO_Workload directory exists: `ls -la ~/GIT/socc22-miso/MISO_Workload/`
- Check shared memory: `ls -la /dev/shm/tmp/MISO_Workload/`
- Re-run: `./workloads/copy_memory.sh`

## Expected Results

After completion, check:
- `logs/mps/JCT.json` - Job Completion Times
- `logs/mps/JRT.json` - Job Running Times
- `logs/mps/QT.json` - Queue Times
- `logs/experiment_mps.log` - Full log

## Next Steps

Once MPS level 33 works, you can test other levels:
```bash
python run_mps_only.py --arrival 100 --num_gpu 1 --num_job 30 --mps_level 50
python run_mps_only.py --arrival 100 --num_gpu 1 --num_job 30 --mps_level 14
```

