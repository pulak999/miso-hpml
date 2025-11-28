# macOS as Client / Ubuntu as GPU Server Setup

## Overview

Yes! You can use macOS as the **client/scheduler** while the Ubuntu machine runs the **GPU server**. This is a common and supported architecture.

```
┌─────────────────┐         TCP/IP          ┌──────────────────┐
│   macOS Client  │  ←──────────────────→   │  Ubuntu GPU      │
│   (Scheduler)   │   Port 10002            │  Server          │
│                 │                          │  (Experiments)   │
│ - Runs exp_mps  │                          │ - Runs gpu_server│
│ - Sends commands│                          │ - Executes jobs  │
│ - No GPU needed │                          │ - Has GPUs       │
└─────────────────┘                          └──────────────────┘
```

## Architecture

- **macOS Client**: Runs the scheduler (`exp_mps.py`, `run_mps_only.py`), sends commands via TCP
- **Ubuntu GPU Server**: Runs `gpu_server.py`, receives commands, executes GPU workloads
- **Communication**: TCP sockets on port 10002 (configurable)

## Setup Steps

### 1. Ubuntu GPU Server Setup (Follow mps_only_plan.md)

On your Ubuntu machine:
1. Install environment: `conda env create -f environment_mps.yml`
2. Set up repository at `/home/${USER}/GIT/socc22-miso`
3. Create MPS scripts (`enable_mps_simple.sh`, etc.)
4. Modify `gpu_server.py` for MPS-only
5. Start GPU server: `python gpu_server.py --node <ubuntu-hostname> --port 10002`

### 2. macOS Client Setup

#### Option A: Minimal Setup (Recommended)

Since the scheduler only needs to send TCP commands, you can set up a minimal Python environment:

```bash
# On macOS
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso

# Create minimal conda environment (no CUDA needed)
conda create -n miso_client python=3.7 -y
conda activate miso_client

# Install only the Python dependencies needed for scheduling
pip install numpy pandas pyyaml psutil

# Or use environment_mps.yml but skip CUDA packages
conda env create -f environment_mps.yml
conda activate tf2
# Remove CUDA packages (they won't work anyway)
pip uninstall torch torchvision torchaudio cudatoolkit cudnn -y
```

#### Option B: Full Environment (If you want to test locally)

Use `environment_mps.yml` with CPU-only PyTorch (for local testing, not GPU experiments).

### 3. Modify Code for macOS Client

The scheduler code has hardcoded Linux paths. You need to modify them:

#### Fix Hardcoded Paths

**File: `exp_mps.py`** (and other experiment files)

Replace:
```python
sys.path.append(f'/home/{user}/GIT/socc22-miso/mps/scheduler/simulator/')
sys.path.append(f'/home/{user}/GIT/socc22-miso/workloads')
```

With:
```python
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f'{REPO_ROOT}/mps/scheduler/simulator/')
sys.path.append(f'{REPO_ROOT}/workloads')
```

Or use an environment variable:
```python
import os
REPO_ROOT = os.environ.get('MISO_REPO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{REPO_ROOT}/mps/scheduler/simulator/')
sys.path.append(f'{REPO_ROOT}/workloads')
```

**File: `controller_helper.py`**

Same fix for hardcoded paths.

**File: `run_mps_only.py`**

Update `physical_nodes` to use the Ubuntu server's hostname/IP:
```python
# Instead of:
hostname = socket.gethostname()
physical_nodes = [hostname]

# Use:
physical_nodes = ['ubuntu-server-hostname']  # or IP address
# Or from environment variable:
physical_nodes = [os.environ.get('GPU_SERVER_HOST', 'ubuntu-server-hostname')]
```

### 4. Network Configuration

#### On Ubuntu Server

1. **Allow TCP connections** on port 10002:
   ```bash
   sudo ufw allow 10002/tcp
   # Or if using iptables:
   sudo iptables -A INPUT -p tcp --dport 10002 -j ACCEPT
   ```

2. **Find server IP/hostname**:
   ```bash
   hostname -I  # Get IP address
   hostname      # Get hostname
   ```

3. **Ensure hostname is resolvable** from macOS:
   - If using hostname, add to `/etc/hosts` on macOS: `sudo nano /etc/hosts`
   - Add: `<ubuntu-ip> ubuntu-server-hostname`
   - Or just use IP address directly

#### On macOS Client

1. **Test connectivity**:
   ```bash
   telnet <ubuntu-ip-or-hostname> 10002
   # Or:
   nc -zv <ubuntu-ip-or-hostname> 10002
   ```

2. **If behind firewall/NAT**, ensure port forwarding is set up.

### 5. Run Experiments

#### On Ubuntu (GPU Server)

```bash
cd ~/GIT/socc22-miso
conda activate tf2
python gpu_server.py --node <ubuntu-hostname> --port 10002
# Keep this running
```

#### On macOS (Client/Scheduler)

```bash
cd ~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso
conda activate miso_client  # or tf2
python run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 30 \
    --random_trace \
    --seed 42 \
    --mps_level 33
```

## Important Considerations

### ✅ What Works

- TCP communication between macOS and Ubuntu
- Scheduler logic (no GPU needed)
- Command sending/receiving
- Results collection

### ⚠️ Potential Issues

1. **Hardcoded Paths**: The codebase has many `/home/${USER}/GIT/socc22-miso` references
   - **Solution**: Modify paths to use relative paths or environment variables

2. **File Paths in Results**: Results are saved locally on the scheduler machine
   - **Solution**: This is fine - results will be on your macOS

3. **Workload Data**: Workload data must be on the **Ubuntu server** (where jobs run)
   - The scheduler just sends commands, but actual model files/data need to be on GPU server

4. **Hostname Resolution**: macOS needs to resolve Ubuntu hostname
   - **Solution**: Use IP address or add to `/etc/hosts`

### 🔧 Quick Path Fix Script

Create a script to fix paths automatically:

```python
# fix_paths.py
import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

files_to_fix = [
    'exp_mps.py',
    'controller_helper.py',
    'exp_full.py',
    'exp_oracle.py',
    'exp_static.py',
    'exp_miso.py',
]

for file in files_to_fix:
    if os.path.exists(file):
        with open(file, 'r') as f:
            content = f.read()
        
        # Replace hardcoded paths
        content = re.sub(
            r"sys\.path\.append\(f'/home/\{user\}/GIT/socc22-miso/([^']+)'\)",
            r"sys.path.append(f'{REPO_ROOT}/\1')",
            content
        )
        
        with open(file, 'w') as f:
            f.write(content)
        print(f"Fixed {file}")
```

## Alternative: Simpler Approach

If path modifications are too complex, you can:

1. **Run scheduler on Ubuntu too** (simpler, no path issues)
   - SSH into Ubuntu
   - Run scheduler there
   - Results will be on Ubuntu

2. **Use SSH port forwarding**:
   ```bash
   # On macOS
   ssh -L 10002:localhost:10002 user@ubuntu-server
   # Then use 'localhost' as the server address
   ```

## Testing the Setup

1. **Test TCP connection**:
   ```bash
   # On macOS
   python -c "from workloads.send_signal import send_signal; send_signal('ubuntu-hostname', 10002, 'test')"
   ```

2. **Check GPU server is listening**:
   ```bash
   # On Ubuntu
   netstat -tuln | grep 10002
   ```

3. **Run a small test**:
   ```bash
   # On macOS
   python run_mps_only.py --num_job 2 --arrival 10
   ```

## Summary

✅ **Yes, macOS can be the client!**

- Set up minimal Python environment on macOS
- Fix hardcoded paths in scheduler code
- Configure network (firewall, hostname resolution)
- Run `gpu_server.py` on Ubuntu
- Run `run_mps_only.py` on macOS
- Results will be saved on macOS

The key is that the scheduler only sends TCP commands - it doesn't need GPU access or CUDA.

