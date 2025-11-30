"""
vLLM Baseline Workload - Continuous Batching Inference
This is a v1 baseline with no sharing, designed to be easily extensible.
"""

import os
import sys
import argparse
import time
import json
import signal
import socket
from pathlib import Path
from datetime import datetime
import subprocess

# Add paths for MISO integration
home = os.environ.get('HOME')
user = os.environ.get('USER')
hostname = socket.gethostname()

# Import vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    print("ERROR: vLLM not installed. Install with: pip install vllm")
    sys.exit(1)

# MISO integration imports
sys.path.append(f'{home}/GIT/socc22-miso/workloads')
from send_signal import send_signal
from checkpoint_helper import CustomCheckpoint

parser = argparse.ArgumentParser(description='vLLM Continuous Batching Inference')
parser.add_argument('--job_id', type=str, default='0', help='ID of job from trace')
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                    help='HuggingFace model name or path')
parser.add_argument('--max_model_len', type=int, default=2048, 
                    help='Maximum sequence length')
parser.add_argument('--tensor_parallel_size', type=int, default=1, 
                    help='Tensor parallelism (number of GPUs for model sharding)')
parser.add_argument('--max_num_seqs', type=int, default=None, 
                    help='Maximum number of sequences in a batch (overrides -b if set)')
parser.add_argument('--max_num_batched_tokens', type=int, default=8192, 
                    help='Maximum number of tokens in a batch (for admission control)')
parser.add_argument('--num_requests', type=int, default=100, 
                    help='Number of inference requests to process')
parser.add_argument('--request_rate', type=float, default=10.0, 
                    help='Requests per second (for arrival simulation)')
parser.add_argument('--max_tokens', type=int, default=512, 
                    help='Maximum tokens to generate per request')
parser.add_argument('--temperature', type=float, default=0.7, 
                    help='Sampling temperature')
parser.add_argument('--top_p', type=float, default=0.9, 
                    help='Top-p sampling')
parser.add_argument('--partition', type=str, default='100', 
                    help='Thread partition percentage (for MPS)')
parser.add_argument('--mps_set', action='store_true', default=False, 
                    help='Enable MPS mode')
parser.add_argument('--port', type=int, default=50051, 
                    help='Port for gRPC communication')
parser.add_argument('--mps_sync', action='store_true', default=False, 
                    help='Require MPS sync start')
parser.add_argument('--iters', type=int, default=200, 
                    help='Number of iterations (requests) to process')
parser.add_argument('--no_ckpt', action='store_true', default=False, 
                    help='Disable checkpointing')
parser.add_argument('--resume', action='store_true', default=False, 
                    help='Resume from checkpoint')
parser.add_argument('--node', type=str, help='Node hostname (scheduler)')
parser.add_argument('--start_batch', type=int, default=0, 
                    help='Starting batch/request number for resume')
parser.add_argument('--cuda_device', type=str, default='0', 
                    help='CUDA device ID when running MPS')

args = parser.parse_args()
save_file = f'/scratch/{user}/mig_ckpt'
starting_request = args.start_batch

# Setup MPS if enabled
if args.mps_set:
    os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = args.partition
    os.environ['CUDA_MPS_LOG_DIRECTORY'] = f'/scratch/{user}/mps_log/nvidia-log-{hostname}/{args.cuda_device}'
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = f'/scratch/{user}/mps_log/nvidia-mps-{hostname}/{args.cuda_device}'

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

# Send PID to scheduler
pid = os.getpid()
message = f'job {args.job_id} pid {pid}'
send_signal(args.node, 10002, message)

# Global state
global start_meas
start_meas = False

# Request tracking
class RequestTracker:
    """Track inference requests and timing"""
    def __init__(self):
        self.request_times = {}
        self.request_begin = 0
        self.request_num = args.start_batch + 1
        self.progress_time = time.time()
        self.completed_requests = 0
        
    def on_request_begin(self, request_id):
        self.request_begin = time.time()
        self.request_times[request_id] = {'start': self.request_begin}
        
    def on_request_end(self, request_id, tokens_generated=0):
        duration = round(time.time() - self.request_begin, 4)
        self.request_times[request_id]['duration'] = duration
        self.request_times[request_id]['tokens'] = tokens_generated
        self.request_num += 1
        self.completed_requests += 1
        
        # Report progress every 10 seconds
        if int(time.time() - self.progress_time) >= 10:
            progress = round((self.completed_requests) / args.iters, 2)
            message = f'job {args.job_id} completion {progress}'
            send_signal(args.node, 10002, message)
            self.progress_time = time.time()
        
        # Check if done
        if self.completed_requests >= args.iters:
            message = f'job {args.job_id} finish'
            send_signal(args.node, 10002, message)
            time.sleep(0.5)
            return True
        return False

tracker = RequestTracker()

# Signal handlers
def terminateProcess(signalNumber, frame):
    """Handle termination signal"""
    sys.exit()

def startProcess(signalNumber, frame):
    """Handle start measurement signal"""
    global start_meas
    start_meas = True

def ckptProcess(signalNumber, frame):
    """Handle checkpoint signal"""
    curr_request = tracker.request_num
    message = f'ckpt job {args.job_id} batch {curr_request}'
    send_signal(args.node, 10002, message)
    time.sleep(0.2)
    sys.exit()

signal.signal(signal.SIGINT, terminateProcess)
signal.signal(signal.SIGCONT, startProcess)
signal.signal(signal.SIGTERM, ckptProcess)

# Initialize vLLM engine
# Use max_num_seqs from args, or fall back to batch_size
max_num_seqs = args.max_num_seqs if args.max_num_seqs is not None else args.batch_size

print(f"Initializing vLLM engine with model: {args.model}")
print(f"Max model length: {args.max_model_len}")
print(f"Max sequences per batch: {max_num_seqs}")
print(f"Max batched tokens: {args.max_num_batched_tokens}")

# Create LLM instance with continuous batching
llm = LLM(
    model=args.model,
    max_model_len=args.max_model_len,
    tensor_parallel_size=args.tensor_parallel_size,
    max_num_seqs=max_num_seqs,
    max_num_batched_tokens=args.max_num_batched_tokens,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
)

# Generate sample prompts (can be customized)
def generate_prompts(num_prompts):
    """Generate sample prompts for inference"""
    base_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the main differences between supervised and unsupervised learning?",
        "Describe how neural networks work.",
        "What is the purpose of backpropagation?",
        "Explain the transformer architecture.",
        "What are attention mechanisms in deep learning?",
        "Describe the difference between RNNs and LSTMs.",
        "What is gradient descent and how does it work?",
        "Explain overfitting and how to prevent it.",
        "What is regularization in machine learning?",
    ]
    
    prompts = []
    for i in range(num_prompts):
        prompt = base_prompts[i % len(base_prompts)]
        prompts.append(f"Request {i}: {prompt}")
    return prompts

# Wait for start signal if needed
if not args.mps_sync:
    print(f"Job {args.job_id} waiting for start signal...")
    while not start_meas:
        time.sleep(0.1)
    print(f"Job {args.job_id} started!")

# Send recovery signal on first request
if starting_request == 0:
    message = f'recover job {args.job_id}'
    send_signal(args.node, 10002, message)
    if args.mps_sync:
        print(f'{args.job_id} sent ready signal')
        # Could add gRPC notification here if needed

# Generate prompts
prompts = generate_prompts(args.iters)

print(f"Starting inference with {args.iters} requests...")
print(f"Request rate: {args.request_rate} req/s")

# Process requests with continuous batching
request_id = 0
start_time = time.time()

try:
    # Process requests in batches to simulate continuous arrival
    batch_size = min(32, max_num_seqs)  # Process in smaller batches
    
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Track batch start
        for i, prompt in enumerate(batch_prompts):
            req_id = batch_start + i
            if req_id >= args.iters:
                break
            tracker.on_request_begin(req_id)
        
        # Run inference (vLLM handles continuous batching automatically)
        if start_meas or not args.mps_sync:
            outputs = llm.generate(batch_prompts, sampling_params)
            
            # Track completion
            for i, output in enumerate(outputs):
                req_id = batch_start + i
                if req_id >= args.iters:
                    break
                tokens_generated = len(output.outputs[0].token_ids) if output.outputs else 0
                done = tracker.on_request_end(req_id, tokens_generated)
                if done:
                    break
        
        # Simulate request arrival rate
        if batch_end < len(prompts):
            time.sleep(1.0 / args.request_rate * batch_size)
        
        if tracker.completed_requests >= args.iters:
            break

except KeyboardInterrupt:
    print(f"\nInterrupted. Completed {tracker.completed_requests}/{args.iters} requests")
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()
finally:
    total_time = time.time() - start_time
    print(f"\nCompleted {tracker.completed_requests} requests in {total_time:.2f} seconds")
    print(f"Average throughput: {tracker.completed_requests / total_time:.2f} req/s")
    
    # Save metrics
    metrics = {
        'total_requests': tracker.completed_requests,
        'total_time': total_time,
        'throughput': tracker.completed_requests / total_time,
        'request_times': tracker.request_times,
    }
    
    log_dir = Path(f'/scratch/{user}/miso_logs/vllm')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / f'job{args.job_id}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {log_dir / f'job{args.job_id}_metrics.json'}")

