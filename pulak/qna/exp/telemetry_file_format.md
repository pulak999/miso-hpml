# GPU Telemetry File Format

## Overview

The integrated DCGM+NVML telemetry collector generates telemetry data in NDJSON (Newline Delimited JSON) format. This format is efficient for streaming large amounts of time-series data.

## File Structure

### GPU-Wide Metrics (`gpu_{id}_gpuwide.ndjson`)

Contains GPU-wide metrics collected via DCGM CLI (`dcgmi dmon`). Each line is a JSON object representing one sample.

**Example line:**
```json
{"timestamp": 1234567890.123, "time_from_start": 0.0, "gpu_id": 0, "GPU Utilization": 85.3, "Memory Utilization": 72.1, "Memory Used": 15234, "Power Draw": 245.2, "Temperature": 72, ...}
```

**Common Fields:**
- `timestamp`: Unix timestamp of sample
- `time_from_start`: Seconds since experiment started
- `gpu_id`: GPU device ID
- `GPU Utilization`: GPU compute utilization (%)
- `Memory Utilization`: Memory bandwidth utilization (%)
- `Memory Used`: GPU memory used (MB)
- `Memory Total`: Total GPU memory (MB)
- `Power Draw`: Power consumption (W)
- `Temperature`: GPU temperature (°C)
- `SM Clock`: Streaming Multiprocessor clock (MHz)
- `Memory Clock`: Memory clock (MHz)
- Additional DCGM fields as available

### Per-Process Metrics (`gpu_{id}_processes.ndjson`)

Contains per-process GPU metrics collected via NVML. Each line is a JSON object with process information.

**Example line:**
```json
{"timestamp": 1234567890.123, "time_from_start": 0.0, "gpu_id": 0, "processes": [{"pid": 12345, "used_memory_mb": 2048.5}, {"pid": 12346, "used_memory_mb": 1024.0}], "num_processes": 2}
```

**Fields:**
- `timestamp`: Unix timestamp of sample
- `time_from_start`: Seconds since experiment started
- `gpu_id`: GPU device ID
- `processes`: Array of process objects
  - `pid`: Process ID
  - `used_memory_mb`: GPU memory used by process (MB)
- `num_processes`: Number of processes using the GPU

### Summary Statistics (`summary_gpu_{id}.json`)

Contains aggregated statistics for all collected metrics. Updated periodically (default: every 60 seconds) and at experiment end.

**Example:**
```json
{
  "gpu_id": 0,
  "final": true,
  "last_updated": 1234567890.123,
  "sample_interval": 1.0,
  "sample_count_estimate": 1031,
  "fields": {
    "GPU Utilization": {
      "count": 1031,
      "mean": 85.3,
      "min": 0.0,
      "max": 100.0,
      "std": 25.4
    },
    "Memory Used": {
      "count": 1031,
      "mean": 15234.5,
      "min": 1024.0,
      "max": 38912.0,
      "std": 8234.2
    },
    "Power Draw": {
      "count": 1031,
      "mean": 245.2,
      "min": 45.0,
      "max": 300.0,
      "std": 52.3
    },
    "process_total_memory_mb": {
      "count": 1031,
      "mean": 3072.5,
      "min": 0.0,
      "max": 8192.0,
      "std": 2048.3
    }
  }
}
```

**Fields:**
- `gpu_id`: GPU device ID
- `final`: Whether this is the final summary (true) or periodic update (false)
- `last_updated`: Timestamp of last update
- `sample_interval`: Sampling interval used
- `sample_count_estimate`: Estimated number of samples collected
- `fields`: Dictionary of metric names to statistics
  - Each statistic includes: `count`, `mean`, `min`, `max`, `std`

## File Rotation

To manage file sizes, the collector automatically rotates files:

1. **Size-based rotation**: When a file exceeds `max_bytes_per_file` (default: 50MB), it's rotated
2. **Timestamped rotation**: Rotated files are renamed with timestamp: `gpu_{id}_gpuwide.{YYYYMMDDTHHMMSS}.ndjson`
3. **Compression**: Rotated files are compressed to `.gz` format (if enabled)
4. **Retention**: Only the most recent `max_rotated_files` (default: 10) rotated files are kept

**Example rotated files:**
```
gpu_0_gpuwide.ndjson                    # Current active file
gpu_0_gpuwide.20241201T143022.ndjson.gz # Rotated file 1
gpu_0_gpuwide.20241201T143045.ndjson.gz # Rotated file 2
...
```

## Reading NDJSON Files

### Python Example

```python
import json

# Read GPU-wide metrics
with open('logs/mps/gpu_0_gpuwide.ndjson') as f:
    for line in f:
        sample = json.loads(line)
        print(f"Time: {sample['time_from_start']:.1f}s, "
              f"GPU Util: {sample.get('GPU Utilization', 'N/A')}%")

# Read per-process metrics
with open('logs/mps/gpu_0_processes.ndjson') as f:
    for line in f:
        sample = json.loads(line)
        print(f"Time: {sample['time_from_start']:.1f}s, "
              f"Processes: {sample['num_processes']}")
        for proc in sample['processes']:
            print(f"  PID {proc['pid']}: {proc['used_memory_mb']:.0f}MB")

# Read summary
with open('logs/mps/summary_gpu_0.json') as f:
    summary = json.load(f)
    for field, stats in summary['fields'].items():
        print(f"{field}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

### Using pandas (for analysis)

```python
import json
import pandas as pd

# Read NDJSON into DataFrame
samples = []
with open('logs/mps/gpu_0_gpuwide.ndjson') as f:
    for line in f:
        samples.append(json.loads(line))

df = pd.DataFrame(samples)

# Analyze
print(df['GPU Utilization'].describe())
print(f"Average power: {df['Power Draw'].mean():.2f}W")
print(f"Peak temperature: {df['Temperature'].max()}°C")

# Plot
import matplotlib.pyplot as plt
plt.plot(df['time_from_start'], df['GPU Utilization'])
plt.xlabel('Time (s)')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization Over Time')
plt.savefig('gpu_util.png')
```

### Reading Compressed Files

```python
import gzip
import json

# Read compressed rotated file
with gzip.open('logs/mps/gpu_0_gpuwide.20241201T143022.ndjson.gz', 'rt') as f:
    for line in f:
        sample = json.loads(line)
        # Process sample...
```

## Advantages of NDJSON Format

1. **Streaming-friendly**: Can process line-by-line without loading entire file
2. **Efficient**: No need to parse entire JSON structure at once
3. **Appendable**: Easy to append new samples
4. **Compressible**: Compresses well (especially with gzip)
5. **Tool support**: Works with many tools (jq, pandas, etc.)

## Comparison with Old Format

**Old format** (`gpu_telemetry_gpu0.json`):
- Single large JSON file with all samples in `timeseries` array
- Required loading entire file into memory
- Summary statistics included in same file

**New format** (NDJSON):
- One JSON object per line
- Can process incrementally
- Separate summary file
- Automatic rotation and compression
- More efficient for large datasets

## Tips

1. **For quick analysis**: Use `summary_gpu_{id}.json` for aggregated statistics
2. **For detailed analysis**: Process NDJSON files line-by-line or load into pandas
3. **For large files**: Use streaming processing to avoid memory issues
4. **For compressed files**: Use `gzip.open()` with `'rt'` mode for text reading

