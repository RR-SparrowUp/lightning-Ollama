#!/bin/bash

# ==============================================================================
# Script Name: cleanup.py
# Description: Stops and cleans up Ollama Docker container and associated resources
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
    """Check if the specified port is in use"""
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
        # Get process IDs using the port
        result = subprocess.run(
            ['lsof', '-t', '-i', f':{port}'],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            # Split PIDs and kill each process
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    pid = pid.strip()
                    if pid:  # Only process non-empty PIDs
                        subprocess.run(['kill', '-9', pid], 
                                     check=True,
                                     stderr=subprocess.DEVNULL)
                        time.sleep(0.5)  # Small delay between kills
                except subprocess.CalledProcessError:
                    continue
            
            # Verify port is now free
            time.sleep(1)  # Wait for processes to be killed
            check_result = subprocess.run(
                ['lsof', '-i', f':{port}'],
                capture_output=True,
                text=True
            )
            return check_result.stdout.strip() == ""
        return True  # Return True if no processes found
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
    
    # Check and clean up both ports
    ports_to_check = [11434, 5000]  # Ollama and Flask ports
    
    for port in ports_to_check:
        port_name = "Ollama" if port == 11434 else "Flask"
        if not check_port_availability(port):
            print_status(f"Cleaning up {port_name} port {port}...")
            if kill_port_process(port):
                print(f"Successfully freed port {port}")
            else:
                print(f"Failed to free port {port}")
        else:
            print(f"Port {port} is already free")
    
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
