#!/bin/bash

# ==============================================================================
# Script Name: pipeline.py
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


import os
import argparse
import toml  
import subprocess
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run sys_info.py with specified configuration.'
    )
    parser.add_argument(
        '--sys_config',
        default='./config.toml',
        help='Path to the system configuration file (default: ./config.toml)'
    )
    return parser.parse_args()

def run_scripts(script_path):
    """Runs various scripts created for specific purposes inside the pipeline"""
    try:
        if not os.path.isfile(script_path):
            print(f"Error: {script_path} does not exist, try cloning from git again")
            return False
        
        print(f"Running {script_path}")
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
        print('Configuration File Generated.')
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def load_system_config(config_path):
    """Load and validate system configuration"""
    if not os.path.exists(config_path):
        print(f"Configuration file does not exist at path: {config_path}")
        print("Running system_info.py to generate configuration file...")
        if not run_scripts('./sys_info.py'):
            return None
    
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def determine_runtime_mode(config):
    """Determine if we should run on CPU or GPU based on system configuration"""
    cuda_info = config['system_info']['CUDA_Info']
    
    if isinstance(cuda_info, str):
        print(f"CUDA Error: {cuda_info}")
        cuda_available = False
    else:
        cuda_available = cuda_info.get('CUDA_Available', False)
    
    cpu_flag = 0 if cuda_available else 1
    gpu_flag = 1 if cuda_available else 0
    
    print("\nRuntime Configuration:")
    print(f"{'GPU Mode' if cuda_available else 'CPU Mode'} will be used")
    print(f"cpu_flag: {cpu_flag}")
    print(f"gpu_flag: {gpu_flag}")
    
    return cpu_flag, gpu_flag

def setup_docker_environment():
    """Set up Docker environment using setup_ollama_docker.sh"""
    print("\nStep 3: Setting up Docker environment")
    
    # Make the script executable
    os.chmod('./setup_ollama_docker.sh', 0o755)
    
    try:
        # Run the setup script
        result = subprocess.run(['./setup_ollama_docker.sh'], 
                              check=True, 
                              text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up Docker environment:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during Docker setup: {e}")
        return False

def check_ollama_container():
    """Check if Ollama container is running"""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=ollama', '--format', '{{.Status}}'],
                              capture_output=True, text=True)
        return 'Up' in result.stdout
    except:
        return False

def wait_for_ollama_ready(timeout=60):
    """Wait for Ollama to be ready to accept connections"""
    print("Waiting for Ollama to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/version'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Ollama is ready!")
                return True
        except:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print("\nTimeout waiting for Ollama to be ready")
    return False

def main():
    args = parse_args()
    
    print("Step 1: Loading system configuration")
    config = load_system_config(args.sys_config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    
    print("\nStep 2: Determining runtime mode")
    cpu_flag, gpu_flag = determine_runtime_mode(config)
    
    print("\nStep 3: Setting up Docker environment")
    if not setup_docker_environment():
        print("Failed to set up Docker environment. Exiting.")
        return
    
    print("\nStep 4: Verifying Ollama container")
    if not check_ollama_container():
        print("Ollama container is not running. Please check the logs.")
        return
    
    print("\nStep 5: Waiting for Ollama to be ready")
    if not wait_for_ollama_ready():
        print("Ollama failed to start properly. Please check the logs.")
        return
    
    print("\nSetup completed successfully!")
    print("Ollama is running on http://localhost:11434")
    print("\nYou can now:")
    print("1. Pull models using: docker exec -it ollama ollama pull <model_name>")
    print("2. Run models using:  docker exec -it ollama ollama run <model_name>")
    print("3. Use the REST API at: http://localhost:11434/api/chat")

if __name__ == "__main__":
    main()

#Saves the system info in config.toml 
#python sys_info.py

#Load config.toml 
#Load the important parameters for running docker 

# with open('./config.toml','r') as f :
#     config = toml.load(f)

# if config['system_info']['CUDA_Info']['CUDA_Available'] = True :
#     cpu_flag = 0
#     gpu_flag = 1  # Run on GPU 
# else :
#     cpu_flag = 1  # Run on CPU
#     gpu_flag = 0   


#Run docker based on the requirements 



#Once docker successfully is installed and returns a flag==1 which states that docker is running


#Rnder UI for downloading the desired model 
#The UI should have the port that we are hosting the downloaded model on 

#Once the model is hosted , pind the model on that port to see if its hosted and running 

#Once true flag is returned .


#We can use the chat.py script to give request and response to the model 


