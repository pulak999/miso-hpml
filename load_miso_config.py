"""
Load MISO unified configuration from miso_config.sh into Python environment.

This module allows Python scripts to access the same configuration values
as the bash scripts, ensuring consistency across the entire system.
"""

import os
import subprocess
import json
from pathlib import Path


def load_miso_config(config_path=None):
    """
    Load MISO configuration from bash config file.
    
    Args:
        config_path: Optional path to config file. If None, searches in:
                    1. MISO_CONFIG_PATH environment variable
                    2. miso_config.sh in repository root
                    3. ~/.miso_config.sh
    
    Returns:
        dict: Configuration values as a dictionary
    """
    # Get repository root
    repo_root = os.environ.get('MISO_REPO_ROOT')
    if not repo_root:
        # Try to get from miso_config.py
        try:
            from miso_config import REPO_ROOT
            repo_root = REPO_ROOT
        except ImportError:
            repo_root = None
    
    # Determine config file path
    if config_path:
        config_file = Path(config_path)
    elif os.environ.get('MISO_CONFIG_PATH'):
        config_file = Path(os.environ.get('MISO_CONFIG_PATH'))
    elif repo_root:
        config_file = Path(repo_root) / 'miso_config.sh'
    else:
        config_file = Path.home() / '.miso_config.sh'
    
    config = {}
    
    # If config file exists, source it and extract variables
    if config_file.exists():
        try:
            # Source the bash config and extract exported variables
            # We'll use a bash subprocess to source it and export to JSON
            bash_cmd = f'''
            source {config_file}
            python3 -c "
            import json
            import os
            config = {{
                'SSH_HOST': os.environ.get('SSH_HOST', ''),
                'REMOTE_IP': os.environ.get('REMOTE_IP', ''),
                'USE_SSH_TUNNEL': os.environ.get('USE_SSH_TUNNEL', 'true'),
                'REMOTE_GPU_SERVER_PORT': os.environ.get('REMOTE_GPU_SERVER_PORT', '10002'),
                'FORWARD_TUNNEL_PORT': os.environ.get('FORWARD_TUNNEL_PORT', '10003'),
                'SCHEDULER_PORT': os.environ.get('SCHEDULER_PORT', '10002'),
                'REVERSE_TUNNEL_REMOTE_PORT': os.environ.get('REVERSE_TUNNEL_REMOTE_PORT', '10002'),
                'CLIENT_REPO_ROOT': os.environ.get('CLIENT_REPO_ROOT', ''),
                'SERVER_REPO_ROOT': os.environ.get('SERVER_REPO_ROOT', ''),
                'SERVER_RESULTS_BASE': os.environ.get('SERVER_RESULTS_BASE', ''),
                'CLIENT_CONDA_ENV': os.environ.get('CLIENT_CONDA_ENV', ''),
                'SERVER_CONDA_ENV': os.environ.get('SERVER_CONDA_ENV', ''),
                'NUM_GPU': os.environ.get('NUM_GPU', '1'),
                'GPU_SERVER_HOST': os.environ.get('GPU_SERVER_HOST', 'localhost'),
                'ARRIVAL': os.environ.get('ARRIVAL', '100'),
                'NUM_JOB': os.environ.get('NUM_JOB', '30'),
                'SEED': os.environ.get('SEED', '42'),
                'COLLECT_TELEMETRY': os.environ.get('COLLECT_TELEMETRY', 'true'),
                'TELEMETRY_INTERVAL': os.environ.get('TELEMETRY_INTERVAL', '1.0'),
            }}
            print(json.dumps(config))
            "
            '''
            result = subprocess.run(
                ['bash', '-c', bash_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                config = json.loads(result.stdout.strip())
        except Exception as e:
            # If loading fails, continue with defaults
            pass
    
    # Apply defaults and type conversions
    defaults = {
        'SSH_HOST': 'l4vm',
        'REMOTE_IP': '172.31.40.254',
        'USE_SSH_TUNNEL': 'true',
        'REMOTE_GPU_SERVER_PORT': '10002',
        'FORWARD_TUNNEL_PORT': '10003',
        'SCHEDULER_PORT': '10002',
        'REVERSE_TUNNEL_REMOTE_PORT': '10002',
        'CLIENT_REPO_ROOT': '',
        'SERVER_REPO_ROOT': '',
        'SERVER_RESULTS_BASE': '',
        'CLIENT_CONDA_ENV': 'miso_client',
        'SERVER_CONDA_ENV': 'tf2',
        'NUM_GPU': '1',
        'GPU_SERVER_HOST': 'localhost',
        'ARRIVAL': '100',
        'NUM_JOB': '30',
        'SEED': '42',
        'COLLECT_TELEMETRY': 'true',
        'TELEMETRY_INTERVAL': '1.0',
    }
    
    # Merge with defaults
    for key, default_value in defaults.items():
        if key not in config or not config[key]:
            config[key] = default_value
    
    # Type conversions
    config['USE_SSH_TUNNEL'] = config['USE_SSH_TUNNEL'].lower() in ('true', '1', 'yes')
    config['COLLECT_TELEMETRY'] = config['COLLECT_TELEMETRY'].lower() in ('true', '1', 'yes')
    config['REMOTE_GPU_SERVER_PORT'] = int(config['REMOTE_GPU_SERVER_PORT'])
    config['FORWARD_TUNNEL_PORT'] = int(config['FORWARD_TUNNEL_PORT'])
    config['SCHEDULER_PORT'] = int(config['SCHEDULER_PORT'])
    config['REVERSE_TUNNEL_REMOTE_PORT'] = int(config['REVERSE_TUNNEL_REMOTE_PORT'])
    config['NUM_GPU'] = int(config['NUM_GPU'])
    config['ARRIVAL'] = int(config['ARRIVAL'])
    config['NUM_JOB'] = int(config['NUM_JOB'])
    config['SEED'] = int(config['SEED'])
    config['TELEMETRY_INTERVAL'] = float(config['TELEMETRY_INTERVAL'])
    
    return config


def get_gpu_server_config():
    """
    Get GPU server configuration (host and port) based on config and SSH tunnel settings.
    
    Returns:
        tuple: (host, port) for GPU server connection
    """
    config = load_miso_config()
    
    if config['USE_SSH_TUNNEL']:
        # Using SSH tunnel - connect to localhost on forward tunnel port
        host = 'localhost'
        port = config['FORWARD_TUNNEL_PORT']
    else:
        # Direct connection - use remote IP and port
        host = config['REMOTE_IP']
        port = config['REMOTE_GPU_SERVER_PORT']
    
    return host, port


# Load config on import (can be overridden by environment variables)
_miso_config = None

def get_config():
    """Get cached config (reloads if MISO_CONFIG_PATH changes)."""
    global _miso_config
    if _miso_config is None or os.environ.get('MISO_CONFIG_PATH'):
        _miso_config = load_miso_config()
    return _miso_config.copy()  # Return copy to prevent modification

