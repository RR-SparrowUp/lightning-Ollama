import os
import argparse
import toml  
import subprocess

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

def main():
    args = parse_args()
    
    print("Step 1: Loading system configuration")
    config = load_system_config(args.sys_config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    
    print("\nStep 2: Determining runtime mode")
    cpu_flag, gpu_flag = determine_runtime_mode(config)
    
    # TODO: Add next steps here
    # print("\nStep 3: Setting up Docker environment")
    # print("Step 4: Downloading and setting up the model")
    # print("Step 5: Starting the chat interface")

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


