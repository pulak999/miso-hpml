# Unified Configuration System

This repository uses a **unified configuration system** that allows you to set up your MPS experiment environment once and use it across all scripts.

## Quick Start

1. **Copy the example config:**
   ```bash
   cp miso_config.sh.example miso_config.sh
   ```

2. **Edit the config with your settings:**
   ```bash
   nano miso_config.sh
   ```

3. **Key things to change:**
   - `SSH_HOST` - Your SSH host alias (e.g., `"l4vm"`, `"hpmlvm-hw1"`)
   - `REMOTE_IP` - Server IP address
   - `CLIENT_REPO_ROOT` - Path on macOS/client
   - `SERVER_REPO_ROOT` - Path on Ubuntu/server
   - `SERVER_RESULTS_BASE` - Where to save results

4. **That's it!** All scripts will automatically use this config.

## Configuration File Location

The system looks for `miso_config.sh` in this order:

1. `MISO_CONFIG_PATH` environment variable (if set)
2. Repository root (`socc22-miso/miso_config.sh`)
3. Home directory (`~/.miso_config.sh`)

## What Gets Configured

The unified config covers:

- ✅ **SSH Connection**: Host, IP, tunnel settings
- ✅ **Ports**: All port configurations (forward, reverse, scheduler, GPU server)
- ✅ **Repository Paths**: Client and server paths
- ✅ **Environment**: Conda environments, modules
- ✅ **GPU Server**: Host, port, number of GPUs
- ✅ **Experiment Parameters**: Jobs, arrival, seed, telemetry
- ✅ **Directories**: Scratch, tmp, logs

## Switching Between Servers

### Option 1: Edit the config file
Just change `SSH_HOST` in `miso_config.sh`:
```bash
# Change from:
SSH_HOST="l4vm"
REMOTE_IP="172.31.40.254"

# To:
SSH_HOST="hpmlvm-hw1"
REMOTE_IP="34.123.45.67"
```

### Option 2: Use different config files
```bash
# Create server-specific configs
cp miso_config.sh miso_config.l4vm.sh
cp miso_config.sh miso_config.hpmlvm.sh

# Edit each with server-specific settings
# Then use the one you want:
export MISO_CONFIG_PATH=$(pwd)/miso_config.l4vm.sh
./setup_ssh_tunnels.sh setup
```

### Option 3: Environment variable overrides
```bash
# Override specific values
export SSH_HOST=hpmlvm-hw1
export REMOTE_IP=34.123.45.67
./setup_ssh_tunnels.sh setup
```

## Scripts That Use This Config

All these scripts automatically load from `miso_config.sh`:

1. **`setup_ssh_tunnels.sh`** - SSH tunnel setup
   - Uses: `SSH_HOST`, `REMOTE_IP`, port settings

2. **`insomnia/sched.sh`** - SLURM job script
   - Uses: Server paths, conda env, modules, experiment params

3. **`insomnia/server_config.sh`** - Server config (loads from unified config)
   - Extends unified config with server-specific overrides

4. **`insomnia/client_config.sh`** - Client config (loads from unified config)
   - Extends unified config with client-specific overrides

5. **`run_mps_only.py`** - Can use environment variables from config
   - Uses: `GPU_SERVER_HOST`, `GPU_SERVER_PORT`, experiment params

## Example: Complete Setup

### Step 1: Create config file
```bash
cd socc22-miso
cp miso_config.sh.example miso_config.sh
nano miso_config.sh
```

### Step 2: Set your server details
```bash
SSH_HOST="l4vm"
REMOTE_IP="172.31.40.254"
CLIENT_REPO_ROOT="~/Desktop/Columbia/Courses/Fall25/HPML/socc22-miso"
SERVER_REPO_ROOT="/home/ubuntu/GIT/socc22-miso"
SERVER_RESULTS_BASE="/insomnia001/home/pm3371/mps_experiments"
```

### Step 3: Use it!
```bash
# Setup SSH tunnels (uses SSH_HOST and REMOTE_IP from config)
./setup_ssh_tunnels.sh setup

# Run experiments (uses ports from config)
python run_mps_only.py --gpu_server_host localhost --gpu_server_port 10003 ...

# Submit SLURM job (uses server paths from config)
sbatch insomnia/sched.sh
```

## Switching to a Different Server

To switch from `l4vm` to `hpmlvm-hw1`:

1. **Update SSH config** (`~/.ssh/config`):
   ```bash
   Host hpmlvm-hw1
     HostName 34.123.45.67
     User pm3371
     ...
   ```

2. **Update `miso_config.sh`**:
   ```bash
   SSH_HOST="hpmlvm-hw1"
   REMOTE_IP="34.123.45.67"  # Get from: ssh hpmlvm-hw1 "hostname -I"
   ```

3. **That's it!** All scripts will now use the new server.

## Verification

Test that your config loads correctly:

```bash
# Load and check config
source miso_config.sh
echo "SSH Host: $SSH_HOST"
echo "Remote IP: $REMOTE_IP"
echo "Client Repo: $CLIENT_REPO_ROOT"
echo "Server Repo: $SERVER_REPO_ROOT"
```

## Troubleshooting

**Config not found:**
- Make sure `miso_config.sh` is in the repository root
- Or set `MISO_CONFIG_PATH` environment variable

**Wrong server:**
- Check `SSH_HOST` matches your `~/.ssh/config` entry
- Verify `REMOTE_IP` is correct: `ssh $SSH_HOST "hostname -I"`

**Port conflicts:**
- Check ports in config match your setup
- Forward tunnel port (10003) should be different from scheduler port (10002)

**Paths wrong:**
- Verify `CLIENT_REPO_ROOT` exists on macOS
- Verify `SERVER_REPO_ROOT` exists on Ubuntu
- Check `SERVER_RESULTS_BASE` is writable

## Files

- `miso_config.sh` - **Your main config file** (create from example)
- `miso_config.sh.example` - Template/example
- `insomnia/server_config.sh` - Server-specific overrides (loads from unified config)
- `insomnia/client_config.sh` - Client-specific overrides (loads from unified config)

## Benefits

✅ **Set up once, use everywhere** - One config file for all scripts  
✅ **Easy server switching** - Change one variable to switch servers  
✅ **No hardcoded values** - Everything is configurable  
✅ **Environment variable overrides** - Override any value when needed  
✅ **Backward compatible** - Scripts still work with defaults if config missing

