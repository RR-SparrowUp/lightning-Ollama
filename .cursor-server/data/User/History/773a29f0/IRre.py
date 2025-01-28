#!/usr/bin/env python3

# ==============================================================================
# Script Name: cleanup.py
# Description: Stops and cleans up Ollama Docker container and associated resources
# Author:      Reiyo
# Version:     1.0.0
# ==============================================================================

import subprocess
import time
import sys
import os

def print_status(message):
    """Print status messages with formatting"""
    print("\n" + "=" * 40)
    print(message)
    print("=" * 40)

def check_docker_running():
    """Check if Docker daemon is running"""
    try:
        subprocess.run(['docker', 'info'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def stop_ollama_container():
    """Stop the Ollama container if it's running"""
    try:
        # Check if container exists
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=ollama', '--format', '{{.Names}}'],
            capture_output=True,
            text=True
        )
        
        if 'ollama' in result.stdout:
            print_status("Stopping Ollama container...")
            
            # Stop the container
            subprocess.run(['docker', 'stop', 'ollama'], 
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            
            # Wait for container to stop
            time.sleep(2)
            
            # Remove the container
            print_status("Removing Ollama container...")
            subprocess.run(['docker', 'rm', '-f', 'ollama'],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error stopping container: {e}")
        return False

def check_port_availability(port=11434):
    """Check if the Ollama port is in use"""
    try:
        result = subprocess.run(
            ['lsof', '-i', f':{port}'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == ""
    except subprocess.CalledProcessError:
        return True

def kill_port_process(port=11434):
    """Kill any process using the specified port"""
    try:
        subprocess.run(
            ['sudo', 'lsof', '-t', '-i', f':{port}', '-sTCP:LISTEN', '|', 'xargs', 'kill', '-9'],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

def remove_ollama_volume():
    """Remove the Ollama Docker volume"""
    try:
        print_status("Removing Ollama volume...")
        subprocess.run(['docker', 'volume', 'rm', '-f', 'ollama'],
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error removing volume: {e}")
        return False

def cleanup():
    """Main cleanup function"""
    print_status("Starting Ollama Cleanup")
    
    # Check if running as root/sudo
    if os.geteuid() != 0:
        print("This script requires sudo privileges to clean up ports.")
        print("Please run with sudo.")
        sys.exit(1)
    
    # Check if Docker is running
    if not check_docker_running():
        print("Docker is not running. Skipping container cleanup.")
        return False
    
    # Stop and remove container
    if stop_ollama_container():
        print("Successfully stopped and removed Ollama container.")
    else:
        print("No Ollama container found or error stopping container.")
    
    # Remove volume
    if remove_ollama_volume():
        print("Successfully removed Ollama volume.")
    else:
        print("No Ollama volume found or error removing volume.")
    
    # Check and clean up port
    if not check_port_availability():
        print_status("Cleaning up port 11434...")
        if kill_port_process():
            print("Successfully freed port 11434")
        else:
            print("Failed to free port 11434")
    else:
        print("Port 11434 is already free")
    
    print_status("Cleanup Complete")
    return True

if __name__ == "__main__":
    try:
        if cleanup():
            print("\nOllama environment has been successfully cleaned up.")
            print("You can now run the setup script again if needed.")
        else:
            print("\nSome cleanup operations failed.")
            print("Please check the messages above for details.")
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during cleanup: {e}")
        sys.exit(1)
