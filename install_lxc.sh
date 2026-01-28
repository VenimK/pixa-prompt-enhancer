#!/bin/bash

# PiXa Prompt Enhancer LXC Installer for Proxmox
# This script creates a Debian LXC container, installs the app, and makes it accessible on the local network.

set -e

# Configuration
CONTAINER_ID=""
CONTAINER_NAME="${2:-pixa-enhancer}"
CPU_CORES=1
RAM_MB=512
DISK_GB=8
BRIDGE="vmbr0"  # Adjust if your bridge is different
SELECTED_STORAGE=""
TEMPLATE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if template exists
check_template() {
    log "Updating template list..."
    pveam update >/dev/null 2>&1

    TEMPLATE=$(pveam available | grep "debian-12" | grep "standard" | tail -n1 | awk '{print $2}')

    if [ -z "$TEMPLATE" ]; then
        error "Could not find Debian 12 standard template."
        log "Available Debian templates:"
        pveam available | grep debian
        exit 1
    fi

    log "Found template: $TEMPLATE"

    if pveam list local | grep -q "$TEMPLATE"; then
        log "Template already exists."
        return 0
    else
        log "Downloading template..."
        if pveam download local "$TEMPLATE"; then
            log "Template downloaded successfully."
            return 0
        else
            error "Failed to download template."
            exit 1
        fi
    fi
}

# Function to get next container ID
get_next_container_id() {
    CONTAINER_ID=$(pvesh get /cluster/nextid)
    log "Using container ID: $CONTAINER_ID"
}

# Function to select storage
select_storage() {
    log "Detecting available storages..."

    # Prefer storages that support rootdir (containers)
    mapfile -t STORAGE_OPTIONS < <(pvesm status -content rootdir 2>/dev/null | awk 'NR>1 {print $1}')

    # Fallback: any storage if none support rootdir
    if [ ${#STORAGE_OPTIONS[@]} -eq 0 ]; then
        mapfile -t STORAGE_OPTIONS < <(pvesm status 2>/dev/null | awk 'NR>1 {print $1}')
    fi

    if [ ${#STORAGE_OPTIONS[@]} -eq 0 ]; then
        error "No storages found. Please configure storage in Proxmox first."
        exit 1
    fi

    echo "Available storage options:"
    idx=1
    for store in "${STORAGE_OPTIONS[@]}"; do
        echo "  ${idx}) ${store}"
        idx=$((idx+1))
    done

    echo
    read -p "Select storage [1-${#STORAGE_OPTIONS[@]}] (default 1): " STORAGE_CHOICE

    if [ -z "${STORAGE_CHOICE}" ]; then
        STORAGE_CHOICE=1
    fi

    if ! [[ "${STORAGE_CHOICE}" =~ ^[0-9]+$ ]] || [ "${STORAGE_CHOICE}" -lt 1 ] || [ "${STORAGE_CHOICE}" -gt ${#STORAGE_OPTIONS[@]} ]; then
        error "Invalid selection."
        exit 1
    fi

    SELECTED_STORAGE=${STORAGE_OPTIONS[$((STORAGE_CHOICE-1))]}
    log "Using storage: $SELECTED_STORAGE"
}

# Function to create LXC container
create_container() {
    log "Creating LXC container $CONTAINER_ID with name '$CONTAINER_NAME'..."

    pct create $CONTAINER_ID local:vztmpl/$TEMPLATE \
        --hostname $CONTAINER_NAME \
        --cores $CPU_CORES \
        --memory $RAM_MB \
        --storage $SELECTED_STORAGE \
        --rootfs $SELECTED_STORAGE:$DISK_GB \
        --net0 name=eth0,bridge=$BRIDGE,ip=dhcp \
        --unprivileged 1 \
        --features nesting=1 \
        --password "changeme123"  # Change this or set interactively

    if [ $? -eq 0 ]; then
        log "Container created successfully."
    else
        error "Failed to create container."
        exit 1
    fi
}

# Function to start container
start_container() {
    log "Starting container $CONTAINER_ID..."
    pct start $CONTAINER_ID
    sleep 10  # Wait for container to fully start
    log "Container started."
}

# Function to install dependencies inside container
install_dependencies() {
    log "Installing dependencies inside container..."

    pct exec $CONTAINER_ID -- bash -c "
        apt update && apt upgrade -y
        apt install -y python3 python3-pip python3-venv git curl

        # Create app directory
        mkdir -p /opt/pixa-enhancer
        cd /opt/pixa-enhancer

        # Clone the repository (replace with your repo URL)
        git clone https://github.com/VenimK/pixa-prompt-enhancer.git .

        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate

        # Install Python dependencies
        pip install -r requirements.txt
        # Install google-generativeai for Linux compatibility
        pip install google-generativeai

        # Create uploads directory
        mkdir -p uploads

        # Create .env file (user needs to add GOOGLE_API_KEY)
        cat > .env << EOF
GOOGLE_API_KEY=your_api_key_here
EOF

        echo 'Installation completed. Please edit /opt/pixa-enhancer/.env and add your GOOGLE_API_KEY.'
    "

    if [ $? -eq 0 ]; then
        log "Dependencies installed successfully."
    else
        error "Failed to install dependencies."
        exit 1
    fi
}

# Function to create systemd service
create_service() {
    log "Creating systemd service for the app..."

    pct exec $CONTAINER_ID -- bash -c "
        cat > /etc/systemd/system/pixa-enhancer.service << EOF
[Unit]
Description=PiXa Prompt Enhancer
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/pixa-enhancer
ExecStart=/opt/pixa-enhancer/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        systemctl enable pixa-enhancer
        systemctl start pixa-enhancer
    "

    if [ $? -eq 0 ]; then
        log "Service created and started."
    else
        error "Failed to create service."
        exit 1
    fi
}

# Function to get container IP
get_container_ip() {
    IP=$(pct exec $CONTAINER_ID -- hostname -I | awk '{print $1}')
    log "Container IP: $IP"
    log "Access the app at: http://$IP:8002"
}

# Main execution
main() {
    log "PiXa Prompt Enhancer LXC Installer"
    get_next_container_id
    log "Container ID: $CONTAINER_ID"
    log "Container Name: $CONTAINER_NAME"
    log "CPU Cores: $CPU_CORES"
    log "RAM: ${RAM_MB}MB"
    log "Disk: ${DISK_GB}GB"

    read -p "Continue with these settings? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Installation cancelled."
        exit 0
    fi

    check_template
    select_storage
    create_container
    start_container
    install_dependencies
    create_service
    get_container_ip

    log "Installation completed successfully!"
    log "Don't forget to:"
    log "1. Change the default password: pct enter $CONTAINER_ID, then passwd"
    log "2. Edit /opt/pixa-enhancer/.env and set your GOOGLE_API_KEY"
    log "3. Access the app at the displayed IP address and port"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run this script as root (sudo)."
    exit 1
fi

# Check if pct command exists
if ! command -v pct &> /dev/null; then
    error "Proxmox pct command not found. Are you running this on a Proxmox host?"
    exit 1
fi

# Run main function
main
