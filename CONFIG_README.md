# MISO Repository Path Configuration

The MISO codebase now uses a configurable repository root path instead of hardcoded `/home/{USER}/GIT/socc22-miso`.

## Quick Start

### Default Behavior (No Configuration Needed)

If your repository is at `/home/{USER}/GIT/socc22-miso`, everything works as before - no changes needed!

### Custom Path Configuration

Set the `MISO_REPO_ROOT` environment variable to use a different path:

```bash
# For your Ubuntu server with path /home/ubuntu/GIT/socc22-miso
export MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso

# Or set it when running commands
MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso python gpu_server.py
```

### Auto-Detection

If you run scripts from within the repository directory, the path is automatically detected (no configuration needed).

## How It Works

The `miso_config.py` module provides:

1. **`REPO_ROOT`**: The repository root path (string)
2. **`get_path(*subpaths)`**: Helper function to build paths relative to repo root

### Priority Order

1. **Environment Variable**: `MISO_REPO_ROOT` (highest priority)
2. **Auto-Detection**: Detects from current file location
3. **Default**: `/home/{USER}/GIT/socc22-miso` (fallback)

## Usage Examples

### In Python Code

```python
from miso_config import REPO_ROOT, get_path

# Get repository root
print(REPO_ROOT)  # /home/ubuntu/GIT/socc22-miso

# Build paths
config_file = get_path('mig_device_autogen.json')
simulator_path = get_path('mps', 'scheduler', 'simulator')
workloads_path = get_path('workloads', 'num_iters.json')
```

### Setting Environment Variable

#### For Single Command
```bash
MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso python gpu_server.py
```

#### For Current Shell Session
```bash
export MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso
python gpu_server.py
python exp_mps.py --num_job 30
```

#### For All Sessions (Permanent)
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso
```

#### In Systemd Service
```ini
[Service]
Environment="MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso"
```

## Updated Files

The following files now use the configurable path:

- `gpu_server.py`
- `exp_mps.py`
- `exp_miso.py`
- `exp_full.py`
- `exp_static.py`
- `exp_oracle.py`
- `controller_helper.py`
- `export_cuda_device_auto.py`
- `dummy/dummy_sender.py`
- `mps/scheduler/simulator/utils.py`

## Migration Guide

### Before (Hardcoded)
```python
sys.path.append(f'/home/{user}/GIT/socc22-miso/mps/scheduler/simulator/')
with open(f'/home/{user}/GIT/socc22-miso/mig_device_autogen.json') as f:
    data = json.load(f)
```

### After (Configurable)
```python
from miso_config import REPO_ROOT, get_path
sys.path.append(get_path('mps', 'scheduler', 'simulator'))
with open(get_path('mig_device_autogen.json')) as f:
    data = json.load(f)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'miso_config'"

**Solution**: Make sure you're running from the repository root, or set `MISO_REPO_ROOT` environment variable.

### Issue: Path not detected correctly

**Solution**: Explicitly set `MISO_REPO_ROOT`:
```bash
export MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso
```

### Issue: Want to verify the path being used

**Solution**: Enable debug mode:
```bash
MISO_DEBUG_CONFIG=1 python your_script.py
```

This will print the repository root path when the config module is imported.

## Benefits

✅ **Portability**: Works on different systems without code changes  
✅ **Flexibility**: Easy to test with different repository locations  
✅ **macOS Client Support**: Enables macOS to act as scheduler/client  
✅ **Backward Compatible**: Default behavior unchanged for existing setups  

## Notes

- The default path (`/home/{USER}/GIT/socc22-miso`) is still used if no environment variable is set
- Auto-detection works when running scripts from within the repository
- All paths are resolved to absolute paths for consistency

