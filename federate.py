import torch
from ultralytics import YOLO
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", default=2, type=int)
args = parser.parse_args()

num_clients = args.num_clients
client_models = {}

# Get local model
# print("Loading local models")
# for i in range(num_clients):
#     client_models[f'{i+1}'] = YOLO(f'client{i+1}/saved_model/yolo-last.pt')


# Evaluate local model against incoming global model - TO BE DONE LATER
# global_metrics = global_model.val()
# local_metrics = local_model.val()
# if global_metrics > local_metrics:
#     model = global_model
# else:
#     model = local_model


# Inference on new images can be skipped for this implementation and use already labelled data eg test set


# Load local models
print("Loading local models")
client_models = {} 

# Write code to get the size of data in the client
# sample_counts = []
# for client_id, (state_dict, count) in enumerate(client_updates):
    # num_samples = len(train_dataset)
    # return trained_model, num_samples

for i in range(num_clients):
    client_models[f'{i+1}'] = YOLO(f'client{i+1}/saved_model/yolo-last.pt')

state_dicts = [client_models[f'{i+1}'].model.state_dict() for i in range(num_clients)]


state_dicts = [client_models[f'{i+1}'].model.state_dict() for i in range(num_clients)]

# Get the set of all keys present in the state dictionaries
all_keys = set(state_dicts[0].keys())
for state_dict in state_dicts:
    all_keys.update(state_dict.keys())

# Step 3: Average the Weights
print("Averaging model weights")
avg_state_dict = {}

# Initialize the averaged state dict with zeros of the same type
for key in all_keys:
    # Check if the key exists in all state_dicts, otherwise skip
    if all(key in state_dict for state_dict in state_dicts):
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
    else:
        # If the key is missing in any state_dict, raise an error or handle as needed
        print(f"Warning: Key {key} is missing in some state dictionaries")

# Sum the weights from each client
for state_dict in state_dicts:
    for key in avg_state_dict.keys():
        avg_state_dict[key] += state_dict[key].float()

# Divide by the number of clients to get the average
for key in avg_state_dict.keys():
    avg_state_dict[key] /= num_clients

# Convert back to original data types
for key in avg_state_dict.keys():
    avg_state_dict[key] = avg_state_dict[key].to(state_dicts[0][key].dtype)

for i in range(num_clients):
    client_models[f'{i+1}'].model.load_state_dict(avg_state_dict)
    # Save the updated model if needed
    print("Save client models")
    client_models[f'{i+1}'].save(f'client{i+1}/saved_model/yolo-avg.pt')
    
    # This line will repeatedly be overwritten to the server. SHOULD BE MODIFIED TO AVOID THIS REPETITION
    print("Save server models")
    client_models[f'{i+1}'].save(f'server/saved_model/yolo-avg.pt') 

#def FedAvg(models):
#    w_avg = copy.deepcopy(models[0])
#    for k in w_avg.keys():
#        for i in range(1, len(models)):
#            w_avg[k] += models[i][k]
#        w_avg[k] = torch.div(w_avg[k], len(models))
#    return w_avg
