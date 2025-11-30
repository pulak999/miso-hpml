# Workload Analysis for MPS LLM Inference Testing

## Where Workloads Are Defined

The MPS experiment system uses a **multi-file configuration** to define which workloads run. Here's where everything is configured:

### 1. Workload Type Mapping: `job_models.json`

**Location**: `socc22-miso/mps/scheduler/simulator/job_models.json`

This file maps job indices (0-99) to workload types. Each entry defines:
- **Model type**: `transformer`, `resnet`, `gnn`, `embedding`, `mobilenet`, `vllm`
- **Batch size**: `16`, `32`, `64`, `128`, `256`, `512`

**Format**: `{model}_train{batch_size}`

**Example**:
```json
{
    "0": "transformer_train32",    // Job 0 → transformer model, batch size 32
    "1": "resnet_train128",        // Job 1 → resnet model, batch size 128
    "2": "transformer_train32",
    "3": "gnn_train512",
    ...
}
```

**How it's used**:
- When a job with ID `N` is scheduled, the system uses `job_models[str(N % 100)]` to get the workload type
- This means jobs 0, 100, 200 all use the same workload type (from index 0)
- The GPU server parses this to extract model name and batch size

### 2. Runtime Durations: `trace_100.json`

**Location**: `socc22-miso/mps/scheduler/trace/trace_100.json`

Maps job indices to expected runtime durations (in seconds):

```json
{
    "0": 120,
    "1": 120,
    "2": 120,
    ...
    "99": 6082.14
}
```

**How it's used**:
- The scheduler uses this to estimate job completion times
- Jobs are selected from this trace when `--random_trace` is used
- Without `--random_trace`, jobs cycle through indices 0-99

### 3. Number of Iterations: `num_iters.json`

**Location**: `socc22-miso/workloads/num_iters.json`

Maps job indices to number of training iterations for each workload:

```json
{
    "0": 100,
    "1": 200,
    "2": 150,
    ...
}
```

**How it's used**:
- When a workload starts, the GPU server looks up `num_iters[mapped_jobid]`
- This determines how many iterations the training script runs
- Different workloads may have different iteration counts

### 4. Workload Execution Scripts

**Location**: `socc22-miso/workloads/`

Python scripts that actually execute the workloads:

- `transformer_train.py` - Transformer model training
- `resnet_train.py` - ResNet image classification
- `gnn_train.py` - Graph Neural Network training
- `embedding_train.py` - Text embedding training
- `mobilenet_train.py` - MobileNet image classification
- `vllm_train.py` - vLLM LLM inference (continuous batching)
- `dummy_train.py` - Minimal test workload

**How they're called**:
```bash
# Example: Job 0 with transformer_train32
python transformer_train.py --job_id 0 -b 32 --iters 100 --node localhost --partition 33 --mps_set --cuda_device 0
```

### 5. How It All Works Together

**Job Selection Flow**:

1. **Job Creation** (`exp_full.py` lines 35-47):
   - Selects job indices (e.g., 0, 1, 2) from `trace_100.json`
   - Gets runtime durations for scheduling

2. **Workload Mapping** (`exp_full.py` line 105-108):
   - Loads `job_models.json`
   - Maps job index → workload type (e.g., job 0 → `"transformer_train32"`)

3. **Execution** (`gpu_server.py` lines 120-125):
   - When job starts, looks up: `job_models[str(jobid % 100)]`
   - Parses workload type: `"transformer_train32"` → model=`transformer`, batch=`32`
   - Gets iterations: `num_iters[str(jobid % 100)]`
   - Executes: `python transformer_train.py --job_id 0 -b 32 --iters 100 ...`

### 6. Modifying Workloads

**To change which workloads run**:

1. **Change workload type for a job index**:
   ```json
   // In job_models.json
   {
       "0": "vllm_train128",  // Changed from "transformer_train32"
       ...
   }
   ```

2. **Change batch size**:
   ```json
   {
       "0": "transformer_train64",  // Changed batch from 32 to 64
       ...
   }
   ```

3. **Change runtime duration**:
   ```json
   // In trace_100.json
   {
       "0": 300,  // Changed from 120 seconds
       ...
   }
   ```

4. **Change iteration count**:
   ```json
   // In num_iters.json
   {
       "0": 500,  // Changed from 100 iterations
       ...
   }
   ```

5. **Add new workload script**:
   - Create new file in `workloads/` directory (e.g., `my_workload_train.py`)
   - Add entries to `job_models.json` (e.g., `"10": "my_workload_train256"`)

**Important Notes**:
- Job IDs are mapped using modulo 100: `jobid % 100` determines which configuration is used
- This means jobs 0, 100, 200 all use the same workload configuration
- The system supports 100 unique workload configurations (indices 0-99)
- All workload scripts must follow the naming convention: `{model}_train.py`
- All scripts must accept standard arguments: `--job_id`, `-b` (batch), `--iters`, `--node`, `--partition`, `--mps_set`, `--cuda_device`

---

## Goal
Test how MPS handles **small to medium LLM inference scenarios** with multi-tenant GPU sharing.

## Workload Assessment

### ✅ **EXCELLENT FIT: vLLM Workload**

**File**: `vllm_train.py`

**Why it's perfect:**
- ✅ **Native LLM inference** - Uses vLLM, the actual production LLM serving framework
- ✅ **Continuous batching** - Matches real-world LLM serving patterns
- ✅ **Variable batch sizes** - Configurable `max_num_seqs` and `max_num_batched_tokens`
- ✅ **Memory-intensive** - LLM models are memory-bound, ideal for MPS testing
- ✅ **Request-based** - Simulates real inference workloads (not training)
- ✅ **Admission control ready** - `max_num_batched_tokens` can be dynamically adjusted

**Characteristics:**
- Model sizes: 7B-13B parameters (small to medium)
- Batch sizes: 128-512 sequences (configurable)
- Token limits: 2048-8192 tokens per batch
- Request patterns: Configurable arrival rates

**MPS Testing Value:**
- **High** - This is the target workload. Perfect for testing:
  - MPS partition allocation (33%, 50%, etc.)
  - Memory isolation between concurrent LLM instances
  - Throughput under different partition schemes
  - Latency impact of sharing

---

### ⚠️ **PARTIAL FIT: Transformer Training**

**File**: `transformer_train.py`

**Why it's partially relevant:**
- ✅ Uses transformer architecture (similar to LLMs)
- ✅ Memory-intensive operations
- ⚠️ **Training workload** (not inference) - Different compute patterns
- ⚠️ **Small model** - Only 4 transformer blocks, head_size=256 (much smaller than LLMs)
- ⚠️ **Fixed batch processing** - No continuous batching
- ⚠️ **Time-series classification** - Different use case than LLM inference

**Characteristics:**
- Model: ~500K parameters (tiny compared to LLMs)
- Batch size: 32-128 (from job_models.json)
- Input: Time-series data (500 timesteps)
- Task: Binary classification

**MPS Testing Value:**
- **Medium** - Can test MPS basics but not LLM-specific patterns
- Useful for: General MPS behavior, memory isolation
- Not useful for: LLM inference patterns, continuous batching, variable sequence lengths

---

### ❌ **POOR FIT: ResNet/MobileNet**

**Files**: `resnet_train.py`, `mobilenet_train.py`

**Why they don't fit:**
- ❌ **Image classification** - Completely different domain
- ❌ **Training workloads** - Not inference
- ❌ **Convolutional operations** - Different memory/compute patterns than transformers
- ❌ **Fixed batch sizes** - No dynamic batching
- ⚠️ **CIFAR-10** - Small images (32x32), not memory-intensive

**Characteristics:**
- ResNet50: ~25M parameters
- MobileNetV2: ~3.4M parameters
- Batch size: 64-512
- Task: Image classification

**MPS Testing Value:**
- **Low** - Different compute patterns, won't reveal LLM-specific MPS issues
- Only useful for: General MPS validation, not LLM inference testing

---

### ❌ **POOR FIT: GNN Training**

**File**: `gnn_train.py`

**Why it doesn't fit:**
- ❌ **Graph neural networks** - Different architecture
- ❌ **Training workload** - Not inference
- ❌ **Molecular property prediction** - Different use case
- ⚠️ **Variable graph sizes** - Interesting but not relevant to LLM inference

**Characteristics:**
- MPNN architecture
- Batch size: 128-512
- Task: Binary classification (molecular properties)

**MPS Testing Value:**
- **Low** - Different patterns, won't help with LLM inference testing

---

### ❌ **POOR FIT: Embedding Training**

**File**: `embedding_train.py`

**Why it doesn't fit:**
- ❌ **Text classification** - Different from LLM inference
- ❌ **Training workload** - Not inference
- ❌ **Small model** - 1D CNN with embeddings, not transformer-based
- ⚠️ **Text processing** - Only similarity is text domain

**Characteristics:**
- 1D CNN with word embeddings
- Batch size: 128-256
- Task: Text classification (20 Newsgroups)

**MPS Testing Value:**
- **Low** - Different architecture and task

---

### ❌ **NOT RELEVANT: Dummy Training**

**File**: `dummy_train.py`

**Why it's not relevant:**
- ❌ **Minimal workload** - Just for testing infrastructure
- ❌ **Fashion-MNIST** - Simple 2-layer Dense network
- ❌ **Not representative** - Too lightweight

**MPS Testing Value:**
- **None** - Only for basic infrastructure testing

---

## Recommendations for LLM Inference MPS Testing

### Primary Workload: vLLM ✅

**Use this for:**
1. **Baseline experiments** - Single LLM instance, no sharing
2. **MPS partition testing** - 33%, 50%, 66% partitions
3. **Multi-tenant scenarios** - 2-3 LLM instances sharing GPU
4. **Admission control** - Testing `max_num_batched_tokens` limits
5. **Throughput vs latency** - Trade-offs under different partitions

**Configuration suggestions:**
```python
# Small LLM (7B) - Good for testing
--model meta-llama/Llama-2-7b-chat-hf
-b 128  # Smaller batches for sharing
--max_num_batched_tokens 4096

# Medium LLM (13B) - More realistic
--model meta-llama/Llama-2-13b-chat-hf  
-b 64   # Even smaller batches
--max_num_batched_tokens 2048
```

### Secondary Workload: Transformer Training ⚠️

**Use this for:**
- **Control experiments** - Compare training vs inference patterns
- **Baseline MPS behavior** - General MPS validation
- **Not for LLM inference testing** - Different patterns

### Don't Use: ResNet, MobileNet, GNN, Embedding, Dummy

These won't provide insights for LLM inference MPS testing.

---

## Key Differences: Training vs Inference

| Aspect | Training Workloads | LLM Inference (vLLM) |
|--------|-------------------|----------------------|
| **Memory pattern** | Forward + backward pass | Forward pass only |
| **Batch processing** | Fixed batches | Continuous batching |
| **Sequence length** | Fixed | Variable (dynamic) |
| **Compute intensity** | High (gradients) | Memory-bound |
| **Latency sensitivity** | Low (offline) | High (online serving) |
| **Request patterns** | Epoch-based | Request-based |

**Implication**: Only vLLM workload accurately represents LLM inference patterns.

---

## MPS Testing Scenarios with vLLM

### Scenario 1: Single LLM Instance (Baseline)
```bash
# No sharing - baseline performance
python vllm_train.py --job_id 0 -b 256 --iters 1000 --mps_set --partition 100
```

### Scenario 2: Two LLM Instances (33% + 33% = 66%)
```bash
# Instance 1
python vllm_train.py --job_id 0 -b 128 --iters 1000 --mps_set --partition 33

# Instance 2  
python vllm_train.py --job_id 1 -b 128 --iters 1000 --mps_set --partition 33
```

### Scenario 3: Three LLM Instances (33% each)
```bash
# Three instances sharing GPU
python vllm_train.py --job_id 0 -b 64 --iters 1000 --mps_set --partition 33
python vllm_train.py --job_id 1 -b 64 --iters 1000 --mps_set --partition 33
python vllm_train.py --job_id 2 -b 64 --iters 1000 --mps_set --partition 33
```

### Scenario 4: Mixed Workloads (LLM + Other)
- **Not recommended** - Other workloads don't match LLM patterns
- Better to test: Multiple LLM instances with different models/sizes

---

## Summary

**For LLM inference MPS testing:**
- ✅ **Use**: `vllm_train.py` - Perfect fit, designed for this
- ⚠️ **Limited use**: `transformer_train.py` - Only for general MPS validation
- ❌ **Don't use**: All other workloads - Different patterns, won't help

**Focus on vLLM workload** - It's the only one that accurately represents small to medium LLM inference scenarios and will provide meaningful insights for MPS multi-tenant GPU sharing.

