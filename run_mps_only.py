#!/usr/bin/env python3
import argparse
import time
import socket
import os
from pathlib import Path
from exp_mps import MPS

# Try to load unified config
try:
    from load_miso_config import get_config, get_gpu_server_config
    _config = get_config()
    _default_host, _default_port = get_gpu_server_config()
except (ImportError, Exception):
    # Fallback if load_miso_config not available or fails
    _config = {}
    _default_host = None
    _default_port = 10003

parser = argparse.ArgumentParser(description='MPS-only experiment')
parser.add_argument('--arrival', type=int, default=_config.get('ARRIVAL', 60), help='inter-arrival period')
parser.add_argument('--num_job', type=int, default=_config.get('NUM_JOB', 30), help='total number of jobs')
parser.add_argument('--num_gpu', type=int, default=_config.get('NUM_GPU', 1), help='total number of GPUs')
parser.add_argument('--seed', type=int, default=_config.get('SEED', 1), help='random seed')
parser.add_argument('--step', type=int, default=10, help='simulation step size')
parser.add_argument('--filler', action='store_true', default=False)
parser.add_argument('--flat_arrival', action='store_true', default=False)
parser.add_argument('--random_trace', action='store_true', default=False)
parser.add_argument('--mps_level', type=int, default=33, help='MPS thread percentage (33, 50, etc.)')
parser.add_argument('--max_tenants', type=int, default=3, help='Maximum number of concurrent tenants (jobs) per GPU (1, 2, or 3)')
parser.add_argument('--collect_telemetry', action='store_true', default=_config.get('COLLECT_TELEMETRY', False), help='Enable GPU telemetry collection using DCGM+NVML')
parser.add_argument('--telemetry_interval', type=float, default=_config.get('TELEMETRY_INTERVAL', 1.0), help='Telemetry sampling interval in seconds (default: 1.0)')
parser.add_argument('--telemetry_max_bytes', type=int, default=50*1024*1024, help='Max bytes per telemetry file before rotation (default: 50MB)')
parser.add_argument('--telemetry_max_files', type=int, default=10, help='Max number of rotated telemetry files to keep (default: 10)')
parser.add_argument('--telemetry_compress', action='store_true', default=True, help='Compress rotated telemetry files (default: True)')
parser.add_argument('--telemetry_summary_interval', type=float, default=60.0, help='Interval for writing summary statistics in seconds (default: 60.0)')
parser.add_argument('--gpu_server_host', type=str, default=_default_host, help='GPU server hostname or IP (use localhost if using SSH tunnel, or set in miso_config.sh)')
parser.add_argument('--gpu_server_port', type=int, default=_default_port, help='GPU server port (default from miso_config.sh, or 10003 with SSH tunnel, 10002 for direct connection)')
parser.add_argument('--error_mean', type=float, default=0.016, help='mean error of predictor')
parser.add_argument('--error_std', type=float, default=0.0032, help='error variance when using Gaussian to generate prediction error')
parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable Weights & Biases (wandb) logging')
parser.add_argument('--wandb_project', type=str, default='mps-experiments', help='W&B project name (default: mps-experiments)')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team name (optional)')
parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (optional, auto-generated if not provided)')
parser.add_argument('--wandb_tags', type=str, nargs='*', default=[], help='W&B tags for this run')
args = parser.parse_args()

# Create logs directory
Path('logs/mps').mkdir(parents=True, exist_ok=True)

# Get GPU server hostname/IP
# If hostname starts with 'ip-', try to extract IP or use localhost for SSH tunnel
def normalize_hostname(hostname):
    """Normalize hostname - if it looks like AWS internal hostname, suggest localhost for SSH tunnel"""
    if hostname and hostname.startswith('ip-'):
        # Extract IP from hostname like 'ip-172-31-40-254' -> '172.31.40.254'
        parts = hostname.split('-')
        if len(parts) >= 4:
            ip = '.'.join(parts[1:])
            print(f'Note: Converted hostname {hostname} to IP {ip}')
            print(f'If using SSH tunnel, use --gpu_server_host localhost instead')
            return ip
    return hostname

# Determine GPU server host and port
if args.gpu_server_host:
    # Use provided host
    normalized_host = normalize_hostname(args.gpu_server_host)
    physical_nodes = [normalized_host]
    print(f'Using GPU server: {normalized_host} (port {args.gpu_server_port})')
elif _default_host:
    # Use from unified config
    normalized_host = normalize_hostname(_default_host)
    physical_nodes = [normalized_host]
    print(f'Using GPU server from config: {normalized_host} (port {args.gpu_server_port})')
    print(f'  (Loaded from miso_config.sh)')
else:
    # Try environment variable as last resort
    gpu_host = os.environ.get('GPU_SERVER_HOST')
    if gpu_host:
        normalized_host = normalize_hostname(gpu_host)
        physical_nodes = [normalized_host]
        print(f'Using GPU server from env: {normalized_host} (port {args.gpu_server_port})')
    else:
        print('ERROR: --gpu_server_host not specified')
        print('Options:')
        print('  1. Set in miso_config.sh (SSH_HOST, REMOTE_IP, USE_SSH_TUNNEL)')
        print('  2. Use --gpu_server_host <host> --gpu_server_port <port>')
        print('  3. Set GPU_SERVER_HOST environment variable')
        print('If using SSH tunnel, use: --gpu_server_host localhost --gpu_server_port 10003')
        exit(1)

print('=' * 60)
print('Running MPS-Only Experiment')
print(f'Jobs: {args.num_job}, GPUs: {args.num_gpu}, MPS Level: {args.mps_level}')
print(f'Max Tenants per GPU: {args.max_tenants}')
if args.collect_telemetry:
    print(f'GPU Telemetry: ENABLED (DCGM+NVML, interval: {args.telemetry_interval}s)')
    print(f'  Output: logs/mps/ (NDJSON format with rotation)')
else:
    print('GPU Telemetry: DISABLED')
if args.use_wandb:
    print(f'W&B Logging: ENABLED (project: {args.wandb_project})')
    if args.wandb_run_name:
        print(f'  Run name: {args.wandb_run_name}')
    if args.wandb_tags:
        print(f'  Tags: {", ".join(args.wandb_tags)}')
else:
    print('W&B Logging: DISABLED')
print(f'GPU Server: {physical_nodes[0]}')
print('=' * 60)

mps_exp = MPS(args, physical_nodes, max_tenants=args.max_tenants)
mps_exp.run(args, mps_lvl=args.mps_level)

print('Experiment completed!')
print(f'Results saved in: logs/mps/')

