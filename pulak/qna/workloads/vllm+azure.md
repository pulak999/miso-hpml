# vLLM + Azure LLM Inference Trace Integration Plan

## Overview
Integrate Azure LLM inference traces (2024 dataset) with vLLM workload to create realistic, production-like inference patterns for MPS testing. Use only 2-3 hours of trace data to keep experiments manageable.

---

## Phase 1: Data Acquisition & Preprocessing

### 1.1 Download Azure Trace Data
- **Source**: Azure Public Dataset
  - Code trace: `AzureLLMInferenceTrace_code_1week.csv`
  - Conversation trace: `AzureLLMInferenceTrace_conv_1week.csv`
- **Location**: Store in `socc22-miso/pulak/qna/workloads/data/azure_traces/`
- **Action**: Download both CSV files using `wget` or `curl`

### 1.2 Data Schema Understanding
The traces contain:
- `TIMESTAMP`: Invocation time (Unix timestamp or datetime)
- `ContextTokens`: Number of input/context tokens
- `GeneratedTokens`: Number of output tokens to generate

### 1.3 Filter to 2-3 Hours of Data
**Strategy**: 
- Parse timestamps and identify a continuous 2-3 hour window
- Choose a window with:
  - **Moderate to high request rate** (good for MPS testing)
  - **Varied token counts** (tests different workload intensities)
  - **Representative patterns** (not just peak or off-peak)

**Implementation**:
```python
# Pseudocode
1. Load CSV files
2. Parse TIMESTAMP column
3. Identify time windows (e.g., hourly chunks)
4. Calculate request rate per hour
5. Select 2-3 hour window with:
   - Request rate: 50-200 req/hour (adjustable)
   - Token distribution: Mix of small (100-500) and large (1000-4000) requests
6. Export filtered trace to: `azure_trace_2hours.csv`
```

**Output**: `azure_trace_2hours.csv` with ~100-600 requests (depending on rate)

---

## Phase 2: Trace-to-vLLM Request Converter

### 2.1 Create Trace Parser Script
**File**: `socc22-miso/pulak/qna/workloads/azure_trace_parser.py`

**Functionality**:
- Parse filtered CSV trace
- Convert to vLLM request format
- Handle timestamp-based arrival simulation
- Generate dummy prompts with correct token counts

**Key Components**:
```python
class AzureTraceParser:
    - load_trace(csv_path)
    - filter_time_window(start_time, duration_hours)
    - normalize_timestamps()  # Convert to relative times
    - generate_requests()  # Convert to vLLM request format
```

### 2.2 Request Format Conversion
**Input (Azure Trace)**:
- `TIMESTAMP`: Absolute time
- `ContextTokens`: Input token count
- `GeneratedTokens`: Output token count

**Output (vLLM Request)**:
- `prompt`: Dummy text with `ContextTokens` tokens
- `max_tokens`: `GeneratedTokens`
- `arrival_time`: Relative timestamp (seconds from start)
- `request_id`: Unique identifier

**Dummy Prompt Generation**:
- Since actual prompt content isn't available (privacy), generate dummy text
- Use tokenizer to ensure exact token count matches `ContextTokens`
- Pattern: `"This is a test prompt. " * N` (adjusted to match token count)

---

## Phase 3: Modify vLLM Workload Script

### 3.1 Add Azure Trace Mode
**File**: `socc22-miso/workloads/vllm_train.py`

**New Arguments**:
```python
--azure_trace: Path to filtered Azure trace CSV
--use_azure_trace: Enable Azure trace mode (overrides synthetic prompts)
--trace_start_offset: Start time offset in seconds (for replay control)
```

### 3.2 Integration Points

**A. Request Generation** (replace `generate_prompts()`):
```python
if args.use_azure_trace:
    requests = load_azure_trace_requests(args.azure_trace)
    # requests = [(arrival_time, prompt, max_tokens), ...]
else:
    requests = generate_prompts(args.iters)
```

**B. Timestamp-Based Arrival** (replace fixed `request_rate`):
```python
# Instead of: time.sleep(1.0 / args.request_rate)
# Use: time.sleep(next_arrival_time - current_time)
```

**C. Dynamic Token Limits**:
```python
# Use GeneratedTokens from trace instead of fixed --max_tokens
sampling_params.max_tokens = request.max_tokens
```

### 3.3 Request Scheduling
**Implementation**:
- Sort requests by `arrival_time`
- Use `time.sleep()` to respect inter-arrival times
- Handle burst arrivals (multiple requests at same timestamp)
- Support time acceleration (e.g., replay 2 hours in 30 minutes)

---

## Phase 4: Testing & Validation

### 4.1 Trace Validation
**Checks**:
- ✅ Request count matches filtered trace
- ✅ Token counts match (context + generated)
- ✅ Arrival times are respected (within tolerance)
- ✅ Total duration matches expected window

### 4.2 vLLM Integration Test
**Test Scenario**:
```bash
# Single instance with Azure trace
python vllm_train.py \
    --use_azure_trace \
    --azure_trace data/azure_trace_2hours.csv \
    --model meta-llama/Llama-2-7b-chat-hf \
    --iters 0  # Auto-determined from trace
```

**Expected Behavior**:
- Loads trace and processes requests
- Respects arrival timestamps
- Generates correct number of tokens
- Logs metrics per request

### 4.3 MPS Testing with Azure Trace
**Scenarios**:
```bash
# Scenario 1: 2 vLLM instances sharing GPU with Azure traces
python vllm_train.py --use_azure_trace --azure_trace trace1.csv --partition 50 --mps_set
python vllm_train.py --use_azure_trace --azure_trace trace2.csv --partition 50 --mps_set

# Scenario 2: Same trace, different partitions
python vllm_train.py --use_azure_trace --azure_trace trace.csv --partition 33 --mps_set
python vllm_train.py --use_azure_trace --azure_trace trace.csv --partition 33 --mps_set
python vllm_train.py --use_azure_trace --azure_trace trace.csv --partition 33 --mps_set
```

---

## Phase 5: Implementation Details

### 5.1 File Structure
```
socc22-miso/pulak/qna/workloads/
├── vllm_train.py              # Modified to support Azure traces
├── azure_trace_parser.py      # New: Trace parsing and conversion
├── data/
│   └── azure_traces/
│       ├── AzureLLMInferenceTrace_code_1week.csv
│       ├── AzureLLMInferenceTrace_conv_1week.csv
│       └── azure_trace_2hours.csv  # Filtered subset
└── vllm+azure.md              # This file
```

### 5.2 Dependencies
**New Requirements**:
- `pandas`: CSV parsing and time window filtering
- `transformers` or `tiktoken`: Token counting for dummy prompt generation
- `datetime`: Timestamp parsing and manipulation

### 5.3 Token Counting Strategy
**Option 1**: Use vLLM's tokenizer (recommended)
- Load model tokenizer: `llm.llm_engine.tokenizer`
- Count tokens: `len(tokenizer.encode(prompt))`
- Adjust prompt length iteratively to match target

**Option 2**: Use `tiktoken` (faster, no model needed)
- Install: `pip install tiktoken`
- Use appropriate encoding (e.g., `cl100k_base` for GPT models)
- Faster for prompt generation, but may not match model exactly

**Recommendation**: Use vLLM's tokenizer for accuracy

---

## Phase 6: Execution Plan

### Step 1: Download & Filter (30 min)
1. Download both CSV files
2. Write filtering script to extract 2-3 hour window
3. Validate filtered data (request count, time range, token distribution)

### Step 2: Build Parser (1-2 hours)
1. Implement `azure_trace_parser.py`
2. Test with filtered trace
3. Validate token counting and prompt generation

### Step 3: Integrate with vLLM (1-2 hours)
1. Modify `vllm_train.py` to support Azure trace mode
2. Implement timestamp-based arrival scheduling
3. Test single-instance execution

### Step 4: MPS Testing (1 hour)
1. Test with 2-3 MPS partitions
2. Compare metrics (throughput, latency) vs synthetic workload
3. Validate realistic behavior

### Step 5: Documentation (30 min)
1. Update usage examples
2. Document trace format requirements
3. Add troubleshooting notes

**Total Estimated Time**: 4-6 hours

---

## Phase 7: Key Considerations

### 7.1 Privacy & Data Handling
- ✅ Azure dataset is public (CC-BY license)
- ✅ No actual prompt content (only token counts)
- ✅ Dummy prompts generated locally
- ✅ Trace data stored locally, not committed to repo

### 7.2 Performance Considerations
- **Time Acceleration**: Option to replay 2 hours in faster time (e.g., 10x speed)
- **Burst Handling**: Multiple requests at same timestamp → batch them
- **Memory**: Large traces → stream processing instead of loading all at once

### 7.3 Trace Selection Criteria
**Good Window Characteristics**:
- Request rate: 50-200 requests/hour (adjustable)
- Token distribution: 
  - ContextTokens: 100-4000 tokens (varied)
  - GeneratedTokens: 50-2000 tokens (varied)
- Time span: Continuous 2-3 hours (no large gaps)

**Avoid**:
- Very sparse periods (< 10 req/hour)
- Extremely dense periods (> 500 req/hour) - may overwhelm single GPU
- Windows with only small requests (not representative)

---

## Phase 8: Expected Outcomes

### 8.1 Realistic Workload Patterns
- ✅ Production-like arrival patterns (not uniform)
- ✅ Varied request sizes (memory and compute stress)
- ✅ Burst arrivals (tests MPS contention)

### 8.2 Better MPS Testing
- ✅ Tests MPS with realistic inference patterns
- ✅ Validates memory isolation under variable loads
- ✅ Measures latency impact of resource sharing

### 8.3 Reproducible Experiments
- ✅ Same trace → same workload (reproducible)
- ✅ Can share filtered traces for consistency
- ✅ Easy to scale (use different time windows)

---

## Phase 9: Future Enhancements

### 9.1 Advanced Features
- **Multiple Traces**: Mix code and conversation traces
- **Trace Replay Modes**: 
  - Real-time (respect original timestamps)
  - Accelerated (10x, 100x speed)
  - Burst mode (all requests at once)
- **Trace Statistics**: Pre-compute and display trace characteristics

### 9.2 Integration Improvements
- **Async API**: Use vLLM's async API for better timestamp accuracy
- **Request Queuing**: Proper queue management for burst arrivals
- **Metrics Alignment**: Match Azure trace metrics with vLLM output

---

## Quick Start (After Implementation)

```bash
# 1. Download traces
cd socc22-miso/pulak/qna/workloads/data/azure_traces/
wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv

# 2. Filter to 2 hours
python filter_trace.py --input AzureLLMInferenceTrace_conv_1week.csv --hours 2 --output azure_trace_2hours.csv

# 3. Run vLLM with Azure trace
python vllm_train.py \
    --use_azure_trace \
    --azure_trace data/azure_traces/azure_trace_2hours.csv \
    --model meta-llama/Llama-2-7b-chat-hf \
    --mps_set \
    --partition 50

# 4. Compare with synthetic workload
python vllm_train.py \
    --num_requests 100 \
    --request_rate 10.0 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --mps_set \
    --partition 50
```

---

## Summary

This plan integrates Azure LLM inference traces with vLLM to create realistic, production-like workloads for MPS testing. The key steps are:

1. **Download & Filter**: Get 2-3 hours of representative trace data
2. **Parse & Convert**: Transform trace format to vLLM requests
3. **Integrate**: Modify vLLM workload to use trace-based requests
4. **Test**: Validate with single and multi-instance MPS scenarios

The result will be a more realistic testing environment that better reflects production LLM inference patterns.

