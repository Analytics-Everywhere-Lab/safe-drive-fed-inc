### CHANGE THIS SCRIPT TO A BASH FILE TO ENABLE ASYNCHRONOUS EXECUTION OF PROGRAMS

### NOW THAT THE DATASET HAS BEEN SPLIT ON THE SERVER. CREATE TWO CLIENT DATASETS
### MODIFY SCRIPT TO RUN FOR EACH OF THE ROUNDS. 
### TO AUTOMATE THE INCREMENTAL PROCESS, INCLUDE A METHOD TO MODIFY THE CONFIG FILES

import os
import yaml
import shutil

def deploy(num_clients):
    server_model_path = 'server/saved_model/yolo-last.pt'
    server_config_path = 'server/config/train.yaml' 

    for i in range(1, num_clients + 1):
            # Dynamically create client paths
            client_model_path = f'client{i}/saved_model/yolo-last.pt'
            client_config_path = f'client{i}/config/train.yaml'
            
            # Ensure the directory for the client's model and config paths exists
            os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(client_config_path), exist_ok=True)
            
            # Copy server model and config to the client paths
            shutil.copy(server_model_path, client_model_path)
            shutil.copy(server_config_path, client_config_path)
            print(f"Deployed {server_model_path} to {client_model_path}")
            print(f"Deployed {server_config_path} to {client_config_path}")

def fine_tune(NUM_CLIENTS, id, if_fed):
    for i in range(NUM_CLIENTS):
        print(f"Fine-tuning Client {i+1}")
        os.system(f"python3 client{i+1}/fine_tune.py --if_fed {if_fed}")
        os.rename("runs/detect/train", f"runs/detect/{id}_fine_tune_client{i+1}")
        print(f'Saved at runs/detect/{id}_fine_tune_client{i+1}')

def federate(NUM_CLIENTS, NUM_ROUNDS, id):
    id = id + '_fed' 
    for i in range(NUM_ROUNDS):
        print(f"Incremental {id}, Round {i} of federated learning")
        os.system("python3 federate.py --num_clients")
        # Fine tune client models in each round
        id = id + str(i+1)       
        fine_tune(NUM_CLIENTS, id=id, if_fed=1)

NUM_CLIENTS = 3

# Init training
print('Training initial round')
os.system("python3 server/scripts/train.py --config config/train.yaml --epochs 1 --image_size 640 --init True")
os.rename("runs/detect/train", "runs/detect/init")
print(f'Saved at runs/detect/init')

deploy(NUM_CLIENTS)

# Fine tune client models - MODIFY TO ALLOW EACH EXECUTE ASYNCHRONOUSLY IF NECESSARY
fine_tune(NUM_CLIENTS, id='init', if_fed=0)

# Federated learning
federate(NUM_CLIENTS, NUM_ROUNDS=1, id='init') #replace with 5

# Incremental learning
rounds = {
    'round_2': "no_seatbelt mobile inattentive",
    'round_3': "no_seatbelt mobile inattentive seatbelt",
    'round_4': "no_seatbelt mobile inattentive seatbelt drowsiness",
    'round_5': "no_seatbelt mobile inattentive seatbelt drowsiness drinking"
}
yaml_file_path = 'server/config/train.yaml'
base_dataset_path = '../../incremental_dataset'

# Incremental Learning
for round_name, classes_list in rounds.items():    
    print(f"Training {round_name}")

    # Train on the incremented dataset
    os.system(f"python3 server/incremental.py --round_name {round_name} --classes_list {classes_list}")
    os.rename(f"runs/detect/train", f"runs/detect/{round_name}")
    print(f'Saved at runs/detect/{round_name}')

    # Deploy to the clients
    deploy(NUM_CLIENTS)

    # Fine-tune client models
    fine_tune(NUM_CLIENTS, id=round_name, if_fed=0)

    # Repeat federated learning
    federate(NUM_CLIENTS, NUM_ROUNDS=1, id=round_name) #replace with 5
