# Configuration Quick Reference

## One-Time Setup

```bash
# 1. Copy example config
cp miso_config.sh.example miso_config.sh

# 2. Edit with your settings
nano miso_config.sh

# 3. Key variables to set:
SSH_HOST="l4vm"                    # Your SSH host alias
REMOTE_IP="172.31.40.254"          # Server IP
CLIENT_REPO_ROOT="~/path/to/repo"  # macOS path
SERVER_REPO_ROOT="/path/to/repo"   # Ubuntu path
```

## Switch Servers

**Change one variable:**
```bash
SSH_HOST="hpmlvm-hw1"  # Change this in miso_config.sh
REMOTE_IP="34.123.45.67"  # Update IP too
```

**Or use environment variable:**
```bash
export SSH_HOST=hpmlvm-hw1
./setup_ssh_tunnels.sh setup
```

## Config File Locations

Scripts look for config in this order:
1. `$MISO_CONFIG_PATH` (environment variable)
2. `miso_config.sh` (repository root)
3. `~/.miso_config.sh` (home directory)

## All Scripts Use This Config

- ✅ `setup_ssh_tunnels.sh` - SSH tunnel setup
- ✅ `insomnia/sched.sh` - SLURM jobs
- ✅ `insomnia/server_config.sh` - Server config
- ✅ `insomnia/client_config.sh` - Client config

## Common Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SSH_HOST` | SSH host alias | `"l4vm"`, `"hpmlvm-hw1"` |
| `REMOTE_IP` | Server IP address | `"172.31.40.254"` |
| `CLIENT_REPO_ROOT` | macOS repo path | `"~/Desktop/.../socc22-miso"` |
| `SERVER_REPO_ROOT` | Ubuntu repo path | `"/home/ubuntu/GIT/socc22-miso"` |
| `SERVER_RESULTS_BASE` | Results directory | `"/insomnia001/home/pm3371/mps_experiments"` |
| `FORWARD_TUNNEL_PORT` | Forward tunnel port | `10003` |
| `SCHEDULER_PORT` | Scheduler listener port | `10002` |

## Verify Config

```bash
source miso_config.sh
echo "SSH Host: $SSH_HOST"
echo "Remote IP: $REMOTE_IP"
```

## Full Documentation

See `README_UNIFIED_CONFIG.md` for complete documentation.

