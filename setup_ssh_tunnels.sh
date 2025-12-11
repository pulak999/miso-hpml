#!/bin/bash
# Setup SSH Tunnels for MPS Experiments
# This script sets up both forward and reverse SSH tunnels between macOS client and Ubuntu GPU server
# Run this from macOS before starting experiments

set -e

# ============================================================
# Load Configuration
# ============================================================
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load unified configuration
# Priority: MISO_CONFIG_PATH env var > script directory > current directory > home directory
if [ -n "$MISO_CONFIG_PATH" ] && [ -f "$MISO_CONFIG_PATH" ]; then
    source "$MISO_CONFIG_PATH"
    echo "Loaded configuration from: $MISO_CONFIG_PATH"
elif [ -f "$SCRIPT_DIR/miso_config.sh" ]; then
    source "$SCRIPT_DIR/miso_config.sh"
    echo "Loaded configuration from: $SCRIPT_DIR/miso_config.sh"
elif [ -f "./miso_config.sh" ]; then
    source "./miso_config.sh"
    echo "Loaded configuration from: ./miso_config.sh"
elif [ -f "$HOME/.miso_config.sh" ]; then
    source "$HOME/.miso_config.sh"
    echo "Loaded configuration from: $HOME/.miso_config.sh"
else
    echo "WARNING: miso_config.sh not found. Using defaults."
    echo "Please create miso_config.sh in the repository root or set MISO_CONFIG_PATH"
    # Fallback to defaults
    SSH_HOST="${SSH_HOST:-l4vm}"
    REMOTE_IP="${REMOTE_IP:-172.31.40.254}"
    FORWARD_LOCAL_PORT="${FORWARD_TUNNEL_PORT:-10003}"
    FORWARD_REMOTE_PORT="${REMOTE_GPU_SERVER_PORT:-10002}"
    REVERSE_REMOTE_PORT="${REVERSE_TUNNEL_REMOTE_PORT:-10002}"
    REVERSE_LOCAL_PORT="${SCHEDULER_PORT:-10002}"
fi

# Map unified config variables to script variables (for backward compatibility)
FORWARD_LOCAL_PORT="${FORWARD_TUNNEL_PORT:-10003}"
FORWARD_REMOTE_PORT="${REMOTE_GPU_SERVER_PORT:-10002}"
REVERSE_REMOTE_PORT="${REVERSE_TUNNEL_REMOTE_PORT:-10002}"
REVERSE_LOCAL_PORT="${SCHEDULER_PORT:-10002}"

# Allow environment variable overrides
SSH_HOST="${SSH_HOST:-l4vm}"
REMOTE_IP="${REMOTE_IP:-172.31.40.254}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SSH Tunnel Setup for MPS Experiments"
echo "=========================================="
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill existing SSH tunnels
kill_existing_tunnels() {
    echo -e "${YELLOW}Checking for existing SSH tunnels...${NC}"
    
    # Kill forward tunnel
    if check_port $FORWARD_LOCAL_PORT; then
        echo -e "${YELLOW}Killing existing forward tunnel on port $FORWARD_LOCAL_PORT...${NC}"
        lsof -ti :$FORWARD_LOCAL_PORT | xargs kill 2>/dev/null || true
        sleep 1
    fi
    
    # Note: Reverse tunnels don't show up in lsof on local side, so we check SSH processes
    echo -e "${YELLOW}Killing existing SSH tunnel processes...${NC}"
    pkill -f "ssh.*-L.*$FORWARD_LOCAL_PORT" 2>/dev/null || true
    pkill -f "ssh.*-R.*$REVERSE_REMOTE_PORT" 2>/dev/null || true
    sleep 2
}

# Function to test SSH connection
test_ssh() {
    echo -e "${YELLOW}Testing SSH connection to $SSH_HOST...${NC}"
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $SSH_HOST "echo 'SSH connection successful'" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH connection successful${NC}"
        return 0
    else
        echo -e "${RED}✗ SSH connection failed${NC}"
        echo -e "${RED}Please check:${NC}"
        echo "  1. SSH host is correct: $SSH_HOST"
        echo "  2. SSH key is set up correctly"
        echo "  3. You can manually connect: ssh $SSH_HOST"
        return 1
    fi
}

# Function to setup forward tunnel
setup_forward_tunnel() {
    echo ""
    echo -e "${YELLOW}Setting up FORWARD tunnel (macOS → Ubuntu)...${NC}"
    echo "  Local port $FORWARD_LOCAL_PORT → Remote $REMOTE_IP:$FORWARD_REMOTE_PORT"
    
    if check_port $FORWARD_LOCAL_PORT; then
        echo -e "${RED}✗ Port $FORWARD_LOCAL_PORT is already in use${NC}"
        echo -e "${YELLOW}  Trying to kill existing process...${NC}"
        kill_existing_tunnels
        sleep 2
    fi
    
    # Set up forward tunnel
    ssh -f -N -L ${FORWARD_LOCAL_PORT}:${REMOTE_IP}:${FORWARD_REMOTE_PORT} $SSH_HOST
    
    sleep 2
    
    if check_port $FORWARD_LOCAL_PORT; then
        echo -e "${GREEN}✓ Forward tunnel established on port $FORWARD_LOCAL_PORT${NC}"
        
        # Test connection
        if nc -zv localhost $FORWARD_LOCAL_PORT 2>&1 | grep -q "succeeded"; then
            echo -e "${GREEN}✓ Forward tunnel connection test successful${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠ Forward tunnel may not be working (GPU server might not be running yet)${NC}"
            return 0
        fi
    else
        echo -e "${RED}✗ Failed to establish forward tunnel${NC}"
        return 1
    fi
}

# Function to setup reverse tunnel
setup_reverse_tunnel() {
    echo ""
    echo -e "${YELLOW}Setting up REVERSE tunnel (Ubuntu → macOS)...${NC}"
    echo "  Remote port $REVERSE_REMOTE_PORT → Local port $REVERSE_LOCAL_PORT"
    echo "  (Allows workloads on Ubuntu to send progress updates to macOS scheduler)"
    
    # Check if local port 10002 is free (needed for scheduler listener)
    if check_port $REVERSE_LOCAL_PORT; then
        PID=$(lsof -ti :$REVERSE_LOCAL_PORT)
        if ps -p $PID -o comm= | grep -q "ssh"; then
            echo -e "${YELLOW}Port $REVERSE_LOCAL_PORT is in use by SSH (likely existing reverse tunnel)${NC}"
            echo -e "${YELLOW}Killing existing reverse tunnel...${NC}"
            kill $PID 2>/dev/null || true
            sleep 2
        else
            echo -e "${RED}✗ Port $REVERSE_LOCAL_PORT is in use by a non-SSH process${NC}"
            echo -e "${RED}  This port is needed for the scheduler listener thread${NC}"
            echo -e "${RED}  Please free this port or stop the conflicting process${NC}"
            echo ""
            echo "Process using port $REVERSE_LOCAL_PORT:"
            lsof -i :$REVERSE_LOCAL_PORT
            return 1
        fi
    fi
    
    # Set up reverse tunnel
    ssh -f -N -R ${REVERSE_REMOTE_PORT}:localhost:${REVERSE_LOCAL_PORT} $SSH_HOST
    
    sleep 2
    
    # Verify reverse tunnel (check SSH process)
    if pgrep -f "ssh.*-R.*$REVERSE_REMOTE_PORT" > /dev/null; then
        echo -e "${GREEN}✓ Reverse tunnel established${NC}"
        echo -e "${GREEN}  Workloads on Ubuntu can now connect to localhost:$REVERSE_REMOTE_PORT${NC}"
        echo -e "${GREEN}  to send progress updates to macOS scheduler${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to establish reverse tunnel${NC}"
        echo -e "${YELLOW}  Note: Reverse tunnels may require GatewayPorts=yes in sshd_config on remote${NC}"
        return 1
    fi
}

# Function to show tunnel status
show_status() {
    echo ""
    echo "=========================================="
    echo "Tunnel Status"
    echo "=========================================="
    
    echo ""
    echo "Forward Tunnel (Commands → GPU Server):"
    if check_port $FORWARD_LOCAL_PORT; then
        echo -e "  ${GREEN}✓ Port $FORWARD_LOCAL_PORT: ACTIVE${NC}"
        lsof -i :$FORWARD_LOCAL_PORT | grep ssh || echo "    (SSH process found)"
    else
        echo -e "  ${RED}✗ Port $FORWARD_LOCAL_PORT: NOT ACTIVE${NC}"
    fi
    
    echo ""
    echo "Reverse Tunnel (Workloads → Scheduler):"
    if pgrep -f "ssh.*-R.*$REVERSE_REMOTE_PORT" > /dev/null; then
        echo -e "  ${GREEN}✓ Reverse tunnel: ACTIVE${NC}"
        ps aux | grep "ssh.*-R.*$REVERSE_REMOTE_PORT" | grep -v grep | head -1
    else
        echo -e "  ${RED}✗ Reverse tunnel: NOT ACTIVE${NC}"
    fi
    
    echo ""
    echo "Scheduler Listener Port:"
    if check_port $REVERSE_LOCAL_PORT; then
        PID=$(lsof -ti :$REVERSE_LOCAL_PORT)
        COMM=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown")
        if [[ "$COMM" == "ssh" ]]; then
            echo -e "  ${GREEN}✓ Port $REVERSE_LOCAL_PORT: Reserved by reverse tunnel${NC}"
        elif [[ "$COMM" == "Python" ]] || [[ "$COMM" == "python" ]] || [[ "$COMM" == "python3" ]]; then
            echo -e "  ${GREEN}✓ Port $REVERSE_LOCAL_PORT: In use by scheduler (experiment running)${NC}"
        else
            echo -e "  ${YELLOW}⚠ Port $REVERSE_LOCAL_PORT: In use by $COMM${NC}"
        fi
    else
        echo -e "  ${GREEN}✓ Port $REVERSE_LOCAL_PORT: FREE (ready for scheduler)${NC}"
    fi
}

# Function to kill all tunnels
kill_tunnels() {
    echo -e "${YELLOW}Killing all SSH tunnels...${NC}"
    pkill -f "ssh.*-L.*$FORWARD_LOCAL_PORT" 2>/dev/null || true
    pkill -f "ssh.*-R.*$REVERSE_REMOTE_PORT" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}✓ All tunnels killed${NC}"
}

# Main execution
main() {
    # Parse arguments
    case "${1:-setup}" in
        setup)
            echo "Setting up SSH tunnels..."
            echo ""
            
            # Test SSH first
            if ! test_ssh; then
                exit 1
            fi
            
            # Kill existing tunnels
            kill_existing_tunnels
            
            # Setup tunnels
            if setup_forward_tunnel && setup_reverse_tunnel; then
                echo ""
                echo -e "${GREEN}=========================================="
                echo -e "✓ All tunnels set up successfully!"
                echo -e "==========================================${NC}"
                show_status
                echo ""
            echo "You can now run experiments with:"
            if [ "$USE_SSH_TUNNEL" = "true" ]; then
                echo "  python run_mps_only.py --gpu_server_host localhost --gpu_server_port $FORWARD_LOCAL_PORT ..."
            else
                echo "  python run_mps_only.py --gpu_server_host $REMOTE_IP --gpu_server_port $FORWARD_REMOTE_PORT ..."
            fi
            else
                echo ""
                echo -e "${RED}=========================================="
                echo -e "✗ Tunnel setup failed"
                echo -e "==========================================${NC}"
                exit 1
            fi
            ;;
        status)
            show_status
            ;;
        kill)
            kill_tunnels
            show_status
            ;;
        test)
            echo "Testing tunnel connections..."
            echo ""
            if check_port $FORWARD_LOCAL_PORT; then
                echo -e "${GREEN}✓ Forward tunnel port $FORWARD_LOCAL_PORT is active${NC}"
                if nc -zv localhost $FORWARD_LOCAL_PORT 2>&1 | grep -q "succeeded"; then
                    echo -e "${GREEN}✓ Forward tunnel connection test: SUCCESS${NC}"
                else
                    echo -e "${YELLOW}⚠ Forward tunnel port active but connection test failed${NC}"
                    echo "  (GPU server might not be running yet)"
                fi
            else
                echo -e "${RED}✗ Forward tunnel port $FORWARD_LOCAL_PORT is not active${NC}"
            fi
            
            echo ""
            if pgrep -f "ssh.*-R.*$REVERSE_REMOTE_PORT" > /dev/null; then
                echo -e "${GREEN}✓ Reverse tunnel is active${NC}"
            else
                echo -e "${RED}✗ Reverse tunnel is not active${NC}"
            fi
            ;;
        *)
            echo "Usage: $0 [setup|status|kill|test]"
            echo ""
            echo "Commands:"
            echo "  setup  - Set up both forward and reverse SSH tunnels (default)"
            echo "  status - Show status of all tunnels"
            echo "  kill   - Kill all SSH tunnels"
            echo "  test   - Test tunnel connections"
            echo ""
            echo "Configuration:"
            echo "  Config file: miso_config.sh (or set MISO_CONFIG_PATH)"
            echo "  Environment variables (override config):"
            echo "    SSH_HOST      - SSH host alias"
            echo "    REMOTE_IP     - Remote server IP"
            echo ""
            echo "Examples:"
            echo "  $0 setup                    # Set up tunnels with defaults"
            echo "  SSH_HOST=myserver $0 setup  # Use custom SSH host"
            echo "  $0 status                  # Check tunnel status"
            echo "  $0 kill                     # Kill all tunnels"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

