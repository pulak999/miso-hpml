# vLLM Baseline Workload

## Overview

This is a **v1 baseline workload** using vLLM for continuous batching inference. It's designed as a baseline with **no sharing** to compare against future multi-tenant implementations.

## Features

- ✅ Continuous batching inference with vLLM
- ✅ MISO integration (signal handling, progress reporting)
- ✅ MPS support (for future multi-tenant experiments)
- ✅ Configurable batch sizes and admission control parameters
- ✅ Easy to extend for scheduler modifications

## Installation

```bash
# Install vLLM (requires CUDA)
pip install vllm

# Or with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Standalone (Baseline Testing)

```bash
# Basic usage
python vllm_train.py \
    --job_id 0 \
    --model meta-llama/Llama-2-7b-chat-hf \
    -b 256 \
    --iters 100 \
    --node localhost

# With custom parameters
python vllm_train.py \
    --job_id 0 \
    --model meta-llama/Llama-2-7b-chat-hf \
    -b 128 \
    --max_num_batched_tokens 4096 \
    --max_tokens 256 \
    --iters 200 \
    --request_rate 20.0 \
    --node localhost
```

### With MPS (Multi-tenant)

```bash
python vllm_train.py \
    --job_id 0 \
    --model meta-llama/Llama-2-7b-chat-hf \
    -b 128 \
    --iters 100 \
    --partition 33 \
    --mps_set \
    --cuda_device 0 \
    --node localhost
```

## Key Parameters

### Batch Size (`-b` or `--batch_size`)
- **Default**: 256
- **Interpretation**: Maps to `max_num_seqs` (maximum sequences in a batch)
- **Usage in job_models.json**: Use format `vllm_train256` where 256 is the batch size

### Admission Control
- `--max_num_batched_tokens`: Maximum tokens in a batch (default: 8192)
- `--max_num_seqs`: Override batch size for max sequences (optional)

### Model Configuration
- `--model`: HuggingFace model name or path
- `--max_model_len`: Maximum sequence length (default: 2048)
- `--tensor_parallel_size`: Number of GPUs for model sharding (default: 1)

### Request Configuration
- `--iters`: Number of inference requests to process (default: 200)
- `--request_rate`: Requests per second for arrival simulation (default: 10.0)
- `--max_tokens`: Maximum tokens to generate per request (default: 512)

## Integration with MISO Scheduler

### Adding to job_models.json

Add vLLM entries to `mps/scheduler/simulator/job_models.json`:

```json
{
    "50": "vllm_train256",
    "51": "vllm_train128",
    "52": "vllm_train512"
}
```

The format is: `vllm_train{BATCH_SIZE}` where `BATCH_SIZE` is the `-b` parameter.

### GPU Server Support

The GPU server (`gpu_server.py`) should automatically handle vLLM jobs when they're in `job_models.json`. The command format is:

```bash
python vllm_train.py --job_id {jobid} -b {batch} --iters {iters} --node {host_node} ...
```

## Extensibility Points (v2+)

### 1. Scheduler Modifications

**Location**: `vllm_train.py` lines ~180-220 (inference loop)

**Future changes**:
- Dynamic `max_num_batched_tokens` based on scheduler decisions
- Admission control integration
- Priority-based request queuing

**Example modification**:
```python
# v2: Dynamic batch token limit from scheduler
max_tokens = get_scheduler_limit()  # From scheduler signal
llm.llm_engine.scheduler_config.max_num_batched_tokens = max_tokens
```

### 2. Multi-tenant Sharing

**Location**: MPS partition handling (already supported)

**Future changes**:
- Per-request MPS partition allocation
- Dynamic partition adjustment based on load

### 3. Checkpointing/Resume

**Location**: Signal handlers and request tracking

**Future changes**:
- Save/restore vLLM engine state
- Resume from specific request number

## Metrics and Logging

Metrics are saved to:
```
/scratch/{user}/miso_logs/vllm/job{job_id}_metrics.json
```

Includes:
- Total requests processed
- Total time
- Throughput (req/s)
- Per-request timing and token counts

## Example Output

```
Initializing vLLM engine with model: meta-llama/Llama-2-7b-chat-hf
Max model length: 2048
Max sequences per batch: 256
Max batched tokens: 8192
Starting inference with 100 requests...
Request rate: 10.0 req/s

Completed 100 requests in 45.23 seconds
Average throughput: 2.21 req/s
Metrics saved to /scratch/user/miso_logs/vllm/job0_metrics.json
```

## Troubleshooting

### vLLM Import Error
```bash
pip install vllm
# Or with CUDA 11.8:
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
- Reduce `--max_num_seqs` (batch size)
- Reduce `--max_num_batched_tokens`
- Reduce `--max_model_len`
- Use smaller model

### Slow Performance
- Increase `--max_num_seqs` for better batching
- Adjust `--request_rate` to match actual arrival rate
- Check GPU utilization with `nvidia-smi`

## Next Steps (v2)

1. **Admission Control**: Integrate scheduler decisions for `max_num_batched_tokens`
2. **Priority Queuing**: Add priority levels to requests
3. **Dynamic Batching**: Adjust batch sizes based on queue depth
4. **Multi-model Support**: Run different models per job
5. **Metrics Collection**: Enhanced telemetry for scheduler optimization

