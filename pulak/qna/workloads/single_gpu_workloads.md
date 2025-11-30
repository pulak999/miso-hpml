# Single-GPU Workload Analysis for MPS Testing

## Context
All workloads in this repository are **single-GPU workloads** by design. They use `CUDA_VISIBLE_DEVICES` to target a specific GPU. MPS (Multi-Process Service) allows multiple such single-GPU processes to **share** a single physical GPU.

## Key Question: Which workloads best test MPS capabilities?

---

## ✅ **EXCELLENT FOR MPS TESTING**

### 1. **vLLM Inference** (`vllm_train.py`)

**Why it's excellent:**
- ✅ **Memory-intensive** - LLM models (7B-13B) are memory-bound, perfect for testing MPS memory isolation
- ✅ **Real-world workload** - Actual production LLM serving scenario
- ✅ **Variable resource usage** - Continuous batching creates variable GPU utilization
- ✅ **Latency-sensitive** - Tests MPS impact on inference latency
- ✅ **Configurable intensity** - Can adjust batch sizes, model sizes, token limits

**MPS Testing Scenarios:**
- **2-3 LLM instances** sharing one GPU (33% + 33% + 33%)
- **Mixed model sizes** (7B + 13B) on same GPU
- **Different batch sizes** to test memory contention
- **Admission control** - Test `max_num_batched_tokens` limits

**Resource Characteristics:**
- Memory: High (model weights + KV cache)
- Compute: Moderate (forward pass only)
- Pattern: Memory-bound, variable batch sizes

**MPS Value: ⭐⭐⭐⭐⭐ (5/5)**

---

### 2. **ResNet Training** (`resnet_train.py`)

**Why it's good:**
- ✅ **Moderate memory usage** - ResNet50 (~25M params) fits on single GPU
- ✅ **Compute-intensive** - Forward + backward pass, good for testing compute sharing
- ✅ **Stable workload** - Consistent batch processing, predictable patterns
- ✅ **Common workload** - Representative of typical single-GPU training

**MPS Testing Scenarios:**
- **2 ResNet instances** (50% + 50% partitions)
- **Mixed with other workloads** - ResNet + MobileNet
- **Different batch sizes** - Test memory vs compute trade-offs

**Resource Characteristics:**
- Memory: Moderate (~2-4GB for ResNet50)
- Compute: High (training with gradients)
- Pattern: Compute-bound, stable batches

**MPS Value: ⭐⭐⭐⭐ (4/5)**

---

### 3. **GNN Training** (`gnn_train.py`)

**Why it's good:**
- ✅ **Variable graph sizes** - Tests MPS with variable workload sizes
- ✅ **Memory patterns** - Graph data structures create interesting memory access patterns
- ✅ **Moderate intensity** - Good middle ground for testing

**MPS Testing Scenarios:**
- **Multiple GNN jobs** with different graph sizes
- **Mixed with other workloads** - GNN + Transformer

**Resource Characteristics:**
- Memory: Moderate-High (depends on graph size)
- Compute: Moderate
- Pattern: Variable, graph-dependent

**MPS Value: ⭐⭐⭐⭐ (4/5)**

---

## ⚠️ **MODERATE FOR MPS TESTING**

### 4. **Transformer Training** (`transformer_train.py`)

**Why it's moderate:**
- ✅ **Transformer architecture** - Relevant for LLM-related testing
- ⚠️ **Small model** - Only ~500K parameters, may not stress MPS enough
- ⚠️ **Time-series task** - Different from typical transformer use cases
- ✅ **Memory patterns** - Still tests attention mechanisms

**MPS Testing Scenarios:**
- **Multiple transformer instances** - But may be too lightweight
- **Mixed workloads** - Transformer + ResNet

**Resource Characteristics:**
- Memory: Low-Moderate (small model)
- Compute: Moderate
- Pattern: Attention-based, but small scale

**MPS Value: ⭐⭐⭐ (3/5)**

---

### 5. **MobileNet Training** (`mobilenet_train.py`)

**Why it's moderate:**
- ✅ **Lightweight model** - MobileNetV2 (~3.4M params)
- ✅ **Efficient architecture** - Tests MPS with efficient models
- ⚠️ **May be too lightweight** - Might not reveal MPS contention issues
- ✅ **Common deployment** - Real-world mobile/edge scenario

**MPS Testing Scenarios:**
- **Multiple MobileNet instances** - Can fit many on one GPU
- **Mixed with heavier workloads** - MobileNet + ResNet

**Resource Characteristics:**
- Memory: Low (~1-2GB)
- Compute: Moderate (efficient operations)
- Pattern: Efficient, lightweight

**MPS Value: ⭐⭐⭐ (3/5)**

---

## ❌ **LIMITED FOR MPS TESTING**

### 6. **Embedding Training** (`embedding_train.py`)

**Why it's limited:**
- ⚠️ **Small model** - 1D CNN with embeddings, not very GPU-intensive
- ⚠️ **Text classification** - Different patterns from typical GPU workloads
- ⚠️ **May not stress MPS** - Too lightweight to reveal issues

**MPS Testing Scenarios:**
- Limited - Better for general validation than stress testing

**Resource Characteristics:**
- Memory: Low
- Compute: Low-Moderate
- Pattern: Text processing, not GPU-intensive

**MPS Value: ⭐⭐ (2/5)**

---

### 7. **Dummy Training** (`dummy_train.py`)

**Why it's limited:**
- ❌ **Minimal workload** - Just for infrastructure testing
- ❌ **Not representative** - Too lightweight for meaningful MPS testing
- ✅ **Good for debugging** - Useful for testing MPS setup

**MPS Testing Scenarios:**
- Only for basic infrastructure validation

**Resource Characteristics:**
- Memory: Very Low
- Compute: Very Low
- Pattern: Minimal

**MPS Value: ⭐ (1/5)**

---

## MPS Testing Recommendations

### **Best Workloads for Comprehensive MPS Testing:**

#### **Tier 1: Primary Testing (High Value)**
1. **vLLM Inference** - Best for LLM inference scenarios
2. **ResNet Training** - Best for general training workloads
3. **GNN Training** - Good for variable workload patterns

#### **Tier 2: Secondary Testing (Moderate Value)**
4. **Transformer Training** - Useful but small scale
5. **MobileNet Training** - Good for lightweight scenarios

#### **Tier 3: Limited Use**
6. **Embedding Training** - Minimal value
7. **Dummy Training** - Infrastructure testing only

---

## MPS Testing Scenarios by Workload Type

### **Scenario 1: Memory-Intensive (LLM Inference)**
```bash
# 2-3 vLLM instances sharing GPU
vllm_train.py --partition 33  # Instance 1
vllm_train.py --partition 33  # Instance 2
vllm_train.py --partition 33  # Instance 3
```
**Best workload:** `vllm_train.py`
**Tests:** Memory isolation, KV cache sharing, latency impact

---

### **Scenario 2: Compute-Intensive (Training)**
```bash
# 2 ResNet instances sharing GPU
resnet_train.py --partition 50  # Instance 1
resnet_train.py --partition 50  # Instance 2
```
**Best workload:** `resnet_train.py`
**Tests:** Compute sharing, gradient computation, batch processing

---

### **Scenario 3: Mixed Workloads**
```bash
# LLM inference + Training
vllm_train.py --partition 50    # Inference
resnet_train.py --partition 50  # Training
```
**Best workloads:** `vllm_train.py` + `resnet_train.py`
**Tests:** Different compute patterns, memory vs compute trade-offs

---

### **Scenario 4: Variable Intensity**
```bash
# Multiple GNN jobs with different graph sizes
gnn_train.py --partition 33  # Small graphs
gnn_train.py --partition 33  # Medium graphs
gnn_train.py --partition 33  # Large graphs
```
**Best workload:** `gnn_train.py`
**Tests:** Variable resource usage, dynamic allocation

---

## Key Metrics to Test with MPS

### **For All Workloads:**
1. **Throughput** - Requests/batches per second
2. **Latency** - P50, P95, P99 percentiles
3. **GPU Utilization** - How well MPS partitions are utilized
4. **Memory Usage** - Peak and average memory per partition
5. **Fairness** - Are partitions getting equal resources?

### **Workload-Specific:**
- **vLLM**: Token generation rate, request completion time
- **ResNet/GNN**: Training speed, epoch time, convergence
- **Mixed**: Interference between different workload types

---

## Summary: Single-GPU Workload Fit for MPS

| Workload | Single-GPU? | MPS Value | Best For |
|----------|-------------|-----------|----------|
| **vLLM** | ✅ Yes | ⭐⭐⭐⭐⭐ | LLM inference, memory-intensive |
| **ResNet** | ✅ Yes | ⭐⭐⭐⭐ | Training, compute-intensive |
| **GNN** | ✅ Yes | ⭐⭐⭐⭐ | Variable workloads |
| **Transformer** | ✅ Yes | ⭐⭐⭐ | Small-scale testing |
| **MobileNet** | ✅ Yes | ⭐⭐⭐ | Lightweight scenarios |
| **Embedding** | ✅ Yes | ⭐⭐ | Limited use |
| **Dummy** | ✅ Yes | ⭐ | Infrastructure only |

**Key Insight:** All workloads are single-GPU, but **vLLM, ResNet, and GNN** provide the best MPS testing value due to their resource intensity and real-world relevance.

---

## Recommended MPS Testing Strategy

1. **Start with vLLM** - Best for LLM inference scenarios
2. **Add ResNet** - For general training workload testing
3. **Mix workloads** - Test different patterns together
4. **Vary partitions** - 33%, 50%, 66% to find optimal splits
5. **Measure everything** - Throughput, latency, utilization, fairness

**Focus on workloads that stress MPS** - Memory-intensive (vLLM) and compute-intensive (ResNet) workloads will reveal MPS behavior better than lightweight ones.

