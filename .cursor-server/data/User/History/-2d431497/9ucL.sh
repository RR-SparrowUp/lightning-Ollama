#!/bin/bash

# ==============================================================================
# Script Name: setup_ollama_docker.sh
# Description: Installs Docker, sets up the NVIDIA Container Toolkit if applicable,
#              and runs the Ollama Docker container based on the system's hardware
#              configuration. It ensures Ollama runs on port 11434 (default Ollama port).
# Author:      Reiyo
# Email:       reiyo@sparrowup.com
# Version:     1.0.0
# Date:        2025-01-17
# License:     MIT License
# ==============================================================================
#
# ==============================================================================
# Copyright (c) 2025 Reiyo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status,treat unset variables as an error, and ensure that pipelines return the exit status of the last command to exit with a non-zero status
set -euo pipefail

# Configuration
CONFIG_FILE="./config.toml"
LOG_FILE="./setup_ollama_docker.log"

# Redirect stdout and stderr to log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Function to print messages
print_message() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to parse configuration TOML file
parse_toml() {
    local key="$1"
    local table="$2"
    if [ -z "$table" ]; then
        grep "^$key" "$CONFIG_FILE" | sed -E 's/^.*= *"?(.*?)"?$/\1/'
    else
        # Extract lines from a specific table
        awk "/^\[$table\]/,/^\[/" "$CONFIG_FILE" | grep "^$key" | sed -E 's/^.*= *"?(.*?)"?$/\1/'
    fi
}

# Function to install Docker and its dependencies
install_docker() {
    if command_exists docker; then
        print_message "Docker is already installed."
    else
        print_message "Installing Docker..."

        sudo apt-get update -y

        sudo apt-get install -y \
            apt-transport-https \
            ca-certificates \
            curl \
            gnupg-agent \
            software-properties-common

        # Add Docker's official GPG key
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

        # Verify the fingerprint 
        sudo apt-key fingerprint 0EBFCD88

        # Add Docker repository
        sudo add-apt-repository \
           "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
           $(lsb_release -cs) \
           stable"

        sudo apt-get update -y

        # Install Docker Engine
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io

        # Add current user to the docker group
        sudo usermod -aG docker "$USER"

        print_message "Docker installed successfully."

        echo "Please log out and log back in to apply Docker group changes."
        exit 0
    fi
}

# Function to install NVIDIA Container Toolkit smf its dependencies based on sys_info
install_nvidia_container_toolkit() {
    print_message "Installing NVIDIA Container Toolkit..."

    # Add the GPG key
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -

    # Add the repository
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu$VERSION_ID/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Create keyring file if it doesn't exist already
    sudo mkdir -p /usr/share/keyrings
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/amd64/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker to use the NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker

    # Restart Docker to apply all the changes 
    sudo systemctl restart docker

    print_message "NVIDIA Container Toolkit installed and configured."
}

# Function to run Ollama Docker container
run_ollama_container() {
    local docker_run_cmd="$1"

    # Check if a container named 'ollama' already exists
    if docker ps -a --format '{{.Names}}' | grep -Eq "^ollama$"; then
        print_message "Existing 'ollama' container found. Removing it..."
        docker stop ollama || true
        docker rm ollama || true
        print_message "'ollama' container removed."
    fi

    print_message "Running Ollama Docker container..."
    eval "$docker_run_cmd"

    print_message "Ollama Docker container is up and running."
}

# Function to check for NVIDIA GPUs
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')
        return 0
    else
        GPU_NAMES=""
        return 1
    fi
}

# Main setup function
setup_ollama() {
    echo "========================================"
    echo "Starting Ollama Docker Setup..."
    echo "========================================"

    # Check if Docker is installed
    if command -v docker &> /dev/null; then
        echo "========================================"
        echo "Docker is already installed."
        echo "========================================"
    else
        # Docker installation code here
        echo "Installing Docker..."
        # ... your Docker installation commands ...
    fi

    # Check Docker group
    if groups $USER | grep -q "docker"; then
        echo "========================================"
        echo "User '$USER' is already in the docker group."
        echo "========================================"
    else
        echo "Adding user to docker group..."
        sudo usermod -aG docker $USER
        echo "User added to docker group. Please log out and back in for changes to take effect."
    fi

    # Check for GPU and set up appropriate Docker configuration
    if check_gpu; then
        echo "========================================"
        echo "NVIDIA GPU(s) detected: $GPU_NAMES"
        echo "Setting up GPU support..."
        echo "========================================"
        
        # GPU-specific setup
        DOCKER_CMD="docker run -d --gpus all --name ollama -v ollama:/root/.ollama -p 11434:11434 ollama/ollama"
    else
        echo "========================================"
        echo "No NVIDIA GPU detected, using CPU mode"
        echo "========================================"
        
        # CPU-only setup
        DOCKER_CMD="docker run -d --name ollama -v ollama:/root/.ollama -p 11434:11434 ollama/ollama"
    fi

    # Run Ollama container
    echo "Starting Ollama container..."
    if $DOCKER_CMD; then
        echo "Ollama container started successfully"
        return 0
    else
        echo "Failed to start Ollama container"
        return 1
    fi
}

# Run the setup
setup_ollama

# ==============================================================================
# End of setup_ollama_docker.sh
# ==============================================================================