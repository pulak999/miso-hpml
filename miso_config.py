"""
MISO Configuration Module

This module provides a centralized way to configure the MISO repository root path.
Set the MISO_REPO_ROOT environment variable to override the default path.

Default: /home/{USER}/GIT/socc22-miso
Override: Set MISO_REPO_ROOT environment variable
"""

import os
from pathlib import Path

# Get repository root from environment variable or use default
def get_repo_root():
    """
    Get the MISO repository root directory.
    
    Priority:
    1. MISO_REPO_ROOT environment variable (if set)
    2. Auto-detect from current file location (if running from repo)
    3. Default: /home/{USER}/GIT/socc22-miso
    
    Returns:
        str: Path to the repository root
    """
    # Check environment variable first
    env_path = os.environ.get('MISO_REPO_ROOT')
    if env_path:
        return os.path.abspath(env_path)
    
    # Try to auto-detect from current file location
    # This file should be at the repo root
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent
    
    # Verify it looks like the repo root (has key directories)
    if (repo_root / 'mps').exists() and (repo_root / 'workloads').exists():
        return str(repo_root)
    
    # Fallback to default
    user = os.environ.get('USER', 'ubuntu')
    default_path = f'/home/{user}/GIT/socc22-miso'
    return default_path

# Set the repository root as a module-level variable
REPO_ROOT = get_repo_root()

def get_path(*subpaths):
    """
    Get a path relative to the repository root.
    
    Args:
        *subpaths: Path components relative to repo root
        
    Returns:
        str: Full path to the requested file/directory
        
    Example:
        get_path('mps', 'scheduler', 'simulator') 
        -> '/home/ubuntu/GIT/socc22-miso/mps/scheduler/simulator'
    """
    return str(Path(REPO_ROOT) / Path(*subpaths))

# Print configuration on import (for debugging)
if __name__ != '__main__':
    # Only print if explicitly enabled via environment variable
    if os.environ.get('MISO_DEBUG_CONFIG', '').lower() in ('1', 'true', 'yes'):
        print(f"[MISO Config] Repository root: {REPO_ROOT}")

