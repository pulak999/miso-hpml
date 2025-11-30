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
print(f'GPU Server: {physical_nodes[0]}')
print('=' * 60)

mps_exp = MPS(args, physical_nodes)
mps_exp.run(args, mps_lvl=args.mps_level)

print('Experiment completed!')
print(f'Results saved in: logs/mps/')

