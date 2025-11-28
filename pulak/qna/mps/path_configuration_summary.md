# Path Configuration Summary

## What Was Changed

All hardcoded paths like `/home/{user}/GIT/socc22-miso` have been replaced with a configurable system using the `miso_config.py` module.

## How to Use

### For Your Ubuntu Server (Path: `/home/ubuntu/GIT/socc22-miso`)

**Option 1: Set Environment Variable (Recommended)**
```bash
export MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso
```

**Option 2: No Configuration Needed**
If the path is exactly `/home/ubuntu/GIT/socc22-miso`, it will work automatically (default behavior).

### Quick Test

```bash
# Verify the path is detected correctly
python -c "from miso_config import REPO_ROOT; print(f'Repo root: {REPO_ROOT}')"
```

## Updated Files

All major files have been updated to use the configurable path:

✅ `gpu_server.py`  
✅ `exp_mps.py`  
✅ `exp_miso.py`  
✅ `exp_full.py`  
✅ `exp_static.py`  
✅ `exp_oracle.py`  
✅ `controller_helper.py`  
✅ `export_cuda_device_auto.py`  
✅ `dummy/dummy_sender.py`  
✅ `mps/scheduler/simulator/utils.py`  

## New Files Created

1. **`miso_config.py`** - Configuration module
2. **`CONFIG_README.md`** - Full documentation

## Benefits

- ✅ No more hardcoded paths
- ✅ Easy to change repository location
- ✅ Works with macOS client setup
- ✅ Backward compatible (default path still works)

## Next Steps

1. Set `MISO_REPO_ROOT` environment variable on your Ubuntu server:
   ```bash
   echo 'export MISO_REPO_ROOT=/home/ubuntu/GIT/socc22-miso' >> ~/.bashrc
   source ~/.bashrc
   ```

2. Test that it works:
   ```bash
   python -c "from miso_config import REPO_ROOT; print(REPO_ROOT)"
   ```

3. Run your experiments as normal - everything should work!

## See Also

- `CONFIG_README.md` - Full documentation
- `macos_client_setup.md` - For macOS client configuration

