#!/usr/bin/env python3
import argparse
import time
import socket
import os
from pathlib import Path
from exp_mps import MPS

parser = argparse.ArgumentParser(description='MPS-only experiment')
parser.add_argument('--arrival', type=int, default=60, help='inter-arrival period')
parser.add_argument('--num_job', type=int, default=30, help='total number of jobs')
parser.add_argument('--num_gpu', type=int, default=1, help='total number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--step', type=int, default=10, help='simulation step size')
parser.add_argument('--filler', action='store_true', default=False)
parser.add_argument('--flat_arrival', action='store_true', default=False)
parser.add_argument('--random_trace', action='store_true', default=False)
parser.add_argument('--mps_level', type=int, default=33, help='MPS thread percentage (33, 50, etc.)')
parser.add_argument('--max_tenants', type=int, default=3, help='Maximum number of concurrent tenants (jobs) per GPU (1, 2, or 3)')
parser.add_argument('--collect_telemetry', action='store_true', default=False, help='Enable GPU telemetry collection using DCGM+NVML')
parser.add_argument('--telemetry_interval', type=float, default=1.0, help='Telemetry sampling interval in seconds (default: 1.0)')
parser.add_argument('--telemetry_max_bytes', type=int, default=50*1024*1024, help='Max bytes per telemetry file before rotation (default: 50MB)')
parser.add_argument('--telemetry_max_files', type=int, default=10, help='Max number of rotated telemetry files to keep (default: 10)')
parser.add_argument('--telemetry_compress', action='store_true', default=True, help='Compress rotated telemetry files (default: True)')
parser.add_argument('--telemetry_summary_interval', type=float, default=60.0, help='Interval for writing summary statistics in seconds (default: 60.0)')
parser.add_argument('--gpu_server_host', type=str, default=None, help='GPU server hostname or IP (use localhost if using SSH tunnel)')
parser.add_argument('--gpu_server_port', type=int, default=10003, help='GPU server port (default: 10003, use 10003 with SSH tunnel, 10002 for direct connection)')
parser.add_argument('--error_mean', type=float, default=0.016, help='mean error of predictor')
parser.add_argument('--error_std', type=float, default=0.0032, help='error variance when using Gaussian to generate prediction error')
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

if args.gpu_server_host:
    # Normalize hostname (convert ip-* format to IP if needed)
    normalized_host = normalize_hostname(args.gpu_server_host)
    physical_nodes = [normalized_host]
    print(f'Using GPU server: {normalized_host} (port {args.gpu_server_port})')
else:
    # Try environment variable, then prompt
    gpu_host = os.environ.get('GPU_SERVER_HOST')
    if gpu_host:
        normalized_host = normalize_hostname(gpu_host)
        physical_nodes = [normalized_host]
        print(f'Using GPU server from env: {normalized_host} (port {args.gpu_server_port})')
    else:
        print('ERROR: --gpu_server_host not specified and GPU_SERVER_HOST env var not set')
        print('Please specify: python run_mps_only.py --gpu_server_host <ubuntu-hostname-or-ip> ...')
        print('If using SSH tunnel, use: --gpu_server_host localhost')
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
print(f'GPU Server: {physical_nodes[0]}')
print('=' * 60)

mps_exp = MPS(args, physical_nodes, max_tenants=args.max_tenants)
mps_exp.run(args, mps_lvl=args.mps_level)

print('Experiment completed!')
print(f'Results saved in: logs/mps/')

