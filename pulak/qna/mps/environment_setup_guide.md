# Environment Setup Guide: Ubuntu vs macOS

## Quick Answer

- **Ubuntu (for GPU experiments)**: Use `environment_mps.yml` ✅
- **macOS**: ⚠️ **CUDA/MPS does NOT work on macOS** - NVIDIA dropped macOS support. You can only set up a development environment (no GPU).

## Detailed Comparison

### `environment.yml` (Original)
- **Contains**: Linux-specific system packages (`_libgcc_mutex`, `ld_impl_linux-64`, `libgcc-ng`, `libstdcxx-ng`, etc.)
- **More complete**: Includes additional system dependencies (ffmpeg, gnutls, etc.)
- **Hardcoded prefix**: Has `prefix: /home/username/anaconda3/envs/tf2` at the end (remove this!)
- **Channels**: Includes `eumetsat` channel
- **Best for**: Full MISO experiments with MIG support (if you have A100 GPUs)

### `environment_mps.yml` (MPS-Only)
- **Contains**: No Linux-specific system packages (more portable)
- **Simpler**: Minimal conda packages, relies on pip for most dependencies
- **No hardcoded prefix**: More portable across systems
- **Channels**: Standard channels only (pytorch, conda-forge, defaults)
- **Best for**: MPS-only experiments (like your Ubuntu L4 setup)

## Recommendations by Platform

### 🐧 Ubuntu (for GPU experiments)

**Use: `environment_mps.yml`**

**Why:**
- Designed for MPS-only experiments (matches your use case)
- No Linux-specific hardcoded packages that might conflict
- Cleaner and more maintainable
- Works perfectly with Ubuntu's package manager

**Setup:**
```bash
conda env create -f environment_mps.yml
conda activate tf2
```

**Note**: If you encounter any missing system libraries, you can install them via `apt`:
```bash
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev
```

### 🍎 macOS (Development Only - NO GPU)

**Use: `environment_mps.yml` (modified)**

**⚠️ CRITICAL LIMITATION**: 
- **CUDA does NOT work on macOS** (NVIDIA dropped support in 2019)
- **MPS (Multi-Process Service) requires CUDA** - will NOT work
- You can only set up the Python environment for code development/testing
- **You CANNOT run GPU experiments on macOS**

**If you want to set up for development only:**

1. Use `environment_mps.yml` as a base
2. Remove CUDA-related packages:
   ```yaml
   # Remove these lines:
   - cudatoolkit=11.3
   - cudnn=8.1
   ```
3. Install CPU-only PyTorch instead:
   ```bash
   conda env create -f environment_mps.yml
   conda activate tf2
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

**Alternative for macOS**: Just install the Python packages you need for development without the full environment.

## Troubleshooting

### Ubuntu Issues

**Issue**: `environment.yml` fails with "package not found"
- **Solution**: Use `environment_mps.yml` instead, or remove the hardcoded `prefix:` line from `environment.yml`

**Issue**: Missing system libraries
- **Solution**: Install via apt: `sudo apt-get install -y build-essential libssl-dev libffi-dev`

### macOS Issues

**Issue**: CUDA packages fail to install
- **Solution**: This is expected! CUDA doesn't work on macOS. Use CPU-only PyTorch.

**Issue**: Linux-specific packages fail
- **Solution**: Use `environment_mps.yml` which doesn't have these packages.

## Summary Table

| Platform | File to Use | GPU Support | Notes |
|----------|-------------|-------------|-------|
| Ubuntu (L4 GPU) | `environment_mps.yml` | ✅ Yes | Best for MPS experiments |
| Ubuntu (A100 MIG) | `environment.yml` | ✅ Yes | For full MISO with MIG |
| macOS (Client) | `environment_mps.yml` (minimal) | ❌ No | **Can act as scheduler/client!** See `macos_client_setup.md` |
| macOS (Dev Only) | `environment_mps.yml` (modified) | ❌ No | Development only, no GPU |

## macOS as Client (Scheduler)

**Great news!** macOS can act as the **client/scheduler** while Ubuntu runs the GPU server. The scheduler only needs to send TCP commands - no GPU required!

**See `macos_client_setup.md` for complete instructions.**

Quick summary:
- ✅ macOS runs scheduler (`exp_mps.py`, `run_mps_only.py`)
- ✅ Ubuntu runs GPU server (`gpu_server.py`)
- ✅ Communication via TCP (port 10002)
- ⚠️ Need to fix hardcoded Linux paths (use `fix_paths_for_macos.py`)

## Next Steps

For your Ubuntu L4 setup:
1. Use `environment_mps.yml`
2. Follow the MPS-only plan in `mps_only_plan.md`
3. You're all set! ✅

