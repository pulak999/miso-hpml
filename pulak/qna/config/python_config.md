# Python Scripts and Unified Configuration

Python scripts can now use the unified configuration system (`miso_config.sh`) through the `load_miso_config.py` module.

## How It Works

1. **Bash scripts** (like `setup_ssh_tunnels.sh`, `sched.sh`) source `miso_config.sh` directly
2. **Python scripts** use `load_miso_config.py` to read the same config file
3. Both get the same values, ensuring consistency

## Updated Python Scripts

### `run_mps_only.py`
- ✅ Loads GPU server host/port from `miso_config.sh`
- ✅ Loads experiment parameters (arrival, num_job, seed, etc.) from config
- ✅ Automatically detects SSH tunnel vs direct connection
- ✅ Falls back to command-line arguments if config not found

### `gpu_server.py`
- ✅ Loads default port from `miso_config.sh`
- ✅ Still accepts `--port` argument to override

## Usage Examples

### Using Config File (Recommended)

1. **Set up `miso_config.sh`:**
   ```bash
   SSH_HOST="l4vm"
   REMOTE_IP="172.31.40.254"
   USE_SSH_TUNNEL=true
   ```

2. **Run experiment (no arguments needed!):**
   ```bash
   python run_mps_only.py --num_job 30 --mps_level 33
   # GPU server host/port automatically loaded from config
   ```

### Overriding Config

You can still override any value:

```bash
# Override GPU server
python run_mps_only.py --gpu_server_host 192.168.1.100 --gpu_server_port 10002

# Override experiment parameters
python run_mps_only.py --num_job 50 --arrival 200
```

### Environment Variables

Environment variables still work and take precedence:

```bash
export GPU_SERVER_HOST=localhost
export GPU_SERVER_PORT=10003
python run_mps_only.py
```

## Priority Order

For `run_mps_only.py`, values are determined in this order:

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Unified config file** (`miso_config.sh`)
4. **Hardcoded defaults** (lowest priority)

## Benefits

✅ **Consistency** - Same config for bash and Python scripts  
✅ **Less typing** - No need to specify `--gpu_server_host` every time  
✅ **Easy switching** - Change server in one place (`miso_config.sh`)  
✅ **Backward compatible** - Still works without config file

## Technical Details

The `load_miso_config.py` module:
- Sources the bash config file using a subprocess
- Extracts exported variables
- Converts types (strings to ints, booleans, etc.)
- Provides helper functions like `get_gpu_server_config()`

## Example: Switching Servers

**Before (without unified config):**
```bash
# Had to specify every time
python run_mps_only.py --gpu_server_host l4vm --gpu_server_port 10003 ...
python run_mps_only.py --gpu_server_host hpmlvm-hw1 --gpu_server_port 10002 ...
```

**After (with unified config):**
```bash
# Edit miso_config.sh once
SSH_HOST="hpmlvm-hw1"
REMOTE_IP="34.123.45.67"
USE_SSH_TUNNEL=false

# Then just run (config automatically used)
python run_mps_only.py --num_job 30 --mps_level 33
```

## Files

- `load_miso_config.py` - Python module to load bash config
- `run_mps_only.py` - Updated to use unified config
- `gpu_server.py` - Updated to use default port from config
- `miso_config.sh` - Unified configuration file

