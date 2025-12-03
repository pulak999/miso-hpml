# GPU Telemetry Collection Setup

## Overview

The MPS experiment framework now includes GPU telemetry collection using NVML (NVIDIA Management Library). This allows you to collect detailed GPU metrics during experiments, including utilization, memory usage, power consumption, temperature, and more.

## Installation

### Prerequisites

Install the `pynvml` Python package:

```bash
pip install nvidia-ml-py
```

Or if using conda:

```bash
conda install -c conda-forge nvidia-ml-py
```

### Verify Installation

```python
import pynvml
pynvml.nvmlInit()
print("NVML initialized successfully")
```

---

## Usage

### Basic Usage

Enable telemetry collection by adding the `--collect_telemetry` flag:

```bash
python3 run_mps_only.py \
    --arrival 100 \
    --num_gpu 1 \
    --num_job 10 \
    --random_trace \
    --seed 42 \
    --mps_level 100 \
    --max_tenants 3 \
    --collect_telemetry \
    --gpu_server_host localhost
```

### Custom Sampling Interval

Change the sampling interval (default: 1.0 seconds):

```bash
python3 run_mps_only.py \
    --collect_telemetry \
    --telemetry_interval 0.5 \
    ...  # other arguments
```

### Disable Process Information

By default, per-process metrics are collected. To disable:

```bash
python3 run_mps_only.py \
    --collect_telemetry \
    --no-telemetry-process-info \
    ...  # other arguments
```

---

## Collected Metrics

The telemetry collector gathers the following metrics for each GPU:

### GPU Metrics
- **GPU Utilization** (`gpu_utilization`): Percentage of GPU compute units in use (0-100%)
- **Memory Utilization** (`memory_utilization`): Percentage of memory bandwidth in use (0-100%)
- **Memory Usage** (`memory_used_mb`, `memory_total_mb`, `memory_free_mb`): Memory statistics in MB
- **Memory Used Percent** (`memory_used_percent`): Percentage of total GPU memory used

### Power and Temperature
- **Power Usage** (`power_usage_watts`): Current power consumption in watts
- **Temperature** (`temperature_c`): GPU temperature in Celsius

### Clock Speeds
- **Graphics Clock** (`graphics_clock_mhz`): Graphics clock speed in MHz
- **SM Clock** (`sm_clock_mhz`): Streaming Multiprocessor clock in MHz
- **Memory Clock** (`memory_clock_mhz`): Memory clock speed in MHz

### Performance State
- **Performance State** (`performance_state`): Current performance state (P0-P12)
- **Throttle Reasons** (`throttle_reasons`): Reasons for clock throttling (if any)

### Process Information (if enabled)
- **Processes** (`processes`): List of processes using the GPU
  - `pid`: Process ID
  - `used_memory_mb`: GPU memory used by process
- **Number of Processes** (`num_processes`): Count of processes using GPU

### Timing
- **Timestamp** (`timestamp`): Unix timestamp of sample
- **Time from Start** (`time_from_start`): Seconds since experiment started

---

## Output Format

Telemetry data is saved to JSON files in the experiment output directory:

**File**: `logs/mps/gpu_telemetry_gpu{N}.json`

Where `N` is the GPU ID (0, 1, 2, ...).

### JSON Structure

```json
{
  "gpu_id": 0,
  "sample_count": 1031,
  "sample_interval": 1.0,
  "summary": {
    "gpu_utilization_mean": 85.3,
    "gpu_utilization_min": 0.0,
    "gpu_utilization_max": 100.0,
    "gpu_utilization_std": 25.4,
    "memory_used_mb_mean": 15234.5,
    "power_usage_watts_mean": 245.2,
    "temperature_c_mean": 72.5,
    ...
  },
  "timeseries": [
    {
      "gpu_id": 0,
      "timestamp": 1234567890.123,
      "time_from_start": 0.0,
      "gpu_utilization": 0,
      "memory_utilization": 0,
      "memory_used_mb": 1024,
      "memory_total_mb": 40960,
      "memory_free_mb": 39936,
      "memory_used_percent": 2.5,
      "power_usage_watts": 45.2,
      "temperature_c": 35,
      "graphics_clock_mhz": 1410,
      "sm_clock_mhz": 1410,
      "memory_clock_mhz": 5001,
      "performance_state": 0,
      "num_processes": 0,
      "processes": []
    },
    ...
  ]
}
```

### Summary Statistics

The `summary` section contains:
- **Mean**: Average value across all samples
- **Min**: Minimum value observed
- **Max**: Maximum value observed
- **Std**: Standard deviation (if > 1 sample)

---

## Integration with Multitenancy Sweep

The telemetry collector automatically works with the multitenancy sweep script. When you run:

```bash
./mps/multitenancy_sweep.sh
```

Each experiment will collect telemetry data if you modify the script to include `--collect_telemetry`. The telemetry files will be saved in each experiment's directory:

- `logs/mps_multitenancy_1tenant/gpu_telemetry_gpu0.json`
- `logs/mps_multitenancy_2tenants/gpu_telemetry_gpu0.json`
- `logs/mps_multitenancy_3tenants/gpu_telemetry_gpu0.json`
- `logs/mps_resource_alloc_mps33/gpu_telemetry_gpu0.json`
- `logs/mps_resource_alloc_mps50/gpu_telemetry_gpu0.json`
- `logs/mps_resource_alloc_mps100/gpu_telemetry_gpu0.json`

---

## Standalone Usage

You can also run the telemetry collector standalone:

```bash
python3 mps/gpu_telemetry.py [gpu_ids] [output_dir] [interval]
```

Examples:

```bash
# Monitor GPU 0, save to logs/mps, sample every 1 second
python3 mps/gpu_telemetry.py 0 logs/mps 1.0

# Monitor GPUs 0 and 1, save to custom directory, sample every 0.5 seconds
python3 mps/gpu_telemetry.py 0,1 /tmp/telemetry 0.5
```

Press Ctrl+C to stop collection.

---

## Analyzing Telemetry Data

### Python Example

```python
import json

# Load telemetry data
with open('logs/mps/gpu_telemetry_gpu0.json') as f:
    data = json.load(f)

# Access summary statistics
summary = data['summary']
print(f"Average GPU utilization: {summary['gpu_utilization_mean']:.2f}%")
print(f"Average power usage: {summary['power_usage_watts_mean']:.2f}W")
print(f"Average temperature: {summary['temperature_c_mean']:.2f}°C")

# Access timeseries data
timeseries = data['timeseries']
for sample in timeseries[:10]:  # First 10 samples
    print(f"Time: {sample['time_from_start']:.1f}s, "
          f"GPU Util: {sample['gpu_utilization']}%, "
          f"Memory: {sample['memory_used_mb']:.0f}MB")
```

### Plotting Example

```python
import json
import matplotlib.pyplot as plt

with open('logs/mps/gpu_telemetry_gpu0.json') as f:
    data = json.load(f)

timeseries = data['timeseries']
time = [s['time_from_start'] for s in timeseries]
gpu_util = [s['gpu_utilization'] for s in timeseries]
power = [s['power_usage_watts'] for s in timeseries if s['power_usage_watts']]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time, gpu_util)
plt.xlabel('Time (s)')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization Over Time')

plt.subplot(1, 2, 2)
plt.plot(time[:len(power)], power)
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.title('Power Consumption Over Time')

plt.tight_layout()
plt.savefig('gpu_telemetry.png')
```

---

## Performance Considerations

- **Sampling Interval**: Lower intervals (e.g., 0.1s) provide more detailed data but:
  - Generate larger files
  - Use more CPU resources
  - May impact experiment performance
  
  Recommended: 1.0s for most experiments, 0.5s for detailed analysis.

- **Process Information**: Collecting per-process metrics adds overhead. Disable if not needed.

- **File Size**: With 1s sampling and a 1000s experiment:
  - ~1000 samples per GPU
  - ~500KB-1MB per JSON file (depending on metrics collected)

---

## Troubleshooting

### NVML Initialization Failed

**Error**: `Failed to initialize NVML`

**Solutions**:
1. Ensure NVIDIA drivers are installed: `nvidia-smi` should work
2. Check permissions: May need to run with appropriate permissions
3. Verify pynvml is installed: `pip list | grep nvidia-ml-py`

### No Data Collected

**Possible Causes**:
1. Telemetry not enabled (missing `--collect_telemetry` flag)
2. Experiment failed before telemetry could start
3. Output directory not writable

**Check**: Look for `gpu_telemetry_gpu0.json` in `logs/mps/`

### Missing Metrics

Some metrics may be `None` if:
- GPU doesn't support the metric (e.g., older GPUs)
- Driver version doesn't support the metric
- Permission issues

This is normal - the collector handles missing metrics gracefully.

---

## Advanced Configuration

### Custom Metrics

To add custom metrics, modify `gpu_telemetry.py` in the `_collect_gpu_metrics` method:

```python
# Example: Add encoder/decoder utilization (for newer GPUs)
try:
    enc_util = pynvml.nvmlDeviceGetEncoderUtilization(handle)
    metrics['encoder_utilization'] = enc_util[0]
    dec_util = pynvml.nvmlDeviceGetDecoderUtilization(handle)
    metrics['decoder_utilization'] = dec_util[0]
except:
    metrics['encoder_utilization'] = None
    metrics['decoder_utilization'] = None
```

---

## Summary

GPU telemetry collection provides detailed insights into GPU behavior during experiments:

- **Resource Usage**: Understand GPU and memory utilization patterns
- **Power Analysis**: Track power consumption and efficiency
- **Thermal Monitoring**: Monitor temperature and throttling
- **Performance Correlation**: Correlate GPU metrics with job performance

Enable it with `--collect_telemetry` and analyze the JSON files to gain deeper insights into your multitenancy experiments!

