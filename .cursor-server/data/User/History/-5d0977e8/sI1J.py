import os
import argparse
import toml  
import subprocess


#Run sys_info.py
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run sys_info.py with specified configuration.'
    )
    parser.add_argument(
        '--sys_config',
        default='./config.toml',
        help='Path to the system configuration file (default: ./config.toml)'
    )
    args = parser.parse_args()
    return args

def run_scripts(script_path):
    """ 
        Runs various scripts created for specefic purposes inside the pipeline 
    
    """
    try:
        if not os.path.isfile(script_path):
            print(f"{script_path} does not exists , try cloning from git again")
            return False
        print(f"Running {script_path}")
        result = subprocess.run(['python',script_path],check=True,capture_output=True,text=True)
        print(result.stdout)
        print('Configuration File Generated.')
        return True
    except subprocess.CalledProcessError as e :
        print(f"An error occured while running {script_path} : ")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpeced error: {e}")
        return False

if __name__ == "__main__":
    args = parse_args()
    fpath = args.sys_config 

    if os.path.exists(fpath):
        with open(fpath, 'r') as f:  
            config = toml.load(f)
            # Add debug print
            print("Raw config content:")
            print(config)
        # Accessing the configuration safely
        try:
            cuda_available = config['system_info']['CUDA_Info']['CUDA_Available']
            print(f"CUDA Available: {cuda_available}")
        except KeyError as e:
            print(f"Configuration key missing: {e}")
    else:
        print(f"Configuration file does not exist at path: {fpath}")
        print(f'Running the system_info.py to generate Configuration file')
        print(f'Running...')
        config_gen = run_scripts('./sys_info.py')

    print("Step 2 : Load config.toml  ")
    with open('./config.toml','r') as f :
        config = toml.load(f)
    if config['system_info']['CUDA_Info']['CUDA_Available'] == True :
        cpu_flag = 0
        gpu_flag = 1  # Run on GPU 
    else :
        cpu_flag = 1  # Run on CPU
        gpu_flag = 0   

    print(f"cpu_flag :{cpu_flag}")
    print(f"gpu_flag :{gpu_flag}")
    

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


