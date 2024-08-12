# A painstaking process of replacing the location of the models and correcting the temp.yaml accordingly
# Can be automated, but for what joy ???

from ultralytics import YOLO
import torch

num_clients = 2
client_models = {}
model1 = YOLO()
model2 = YOLO()
# model_avg = YOLO()
server = YOLO(f'runs/detect/round_2/weights/best.pt')
# model1 = YOLO(f'runs/detect/round_3_fed12345_fine_tune_client1/weights/best.pt')
# model2 = YOLO(f'runs/detect/round_3_fed12345_fine_tune_client2/weights/best.pt')

client_models = {}
for i in range(num_clients):
    client_models[f'{i+1}'] = YOLO(f'runs/detect/round_2_fed12345_fine_tune_client{i+1}/weights/best.pt')

state_dicts = [client_models[f'{i+1}'].model.state_dict() for i in range(num_clients)]


state_dicts = [client_models[f'{i+1}'].model.state_dict() for i in range(num_clients)]

# Get the set of all keys present in the state dictionaries
all_keys = set(state_dicts[0].keys())
for state_dict in state_dicts:
    all_keys.update(state_dict.keys())

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

client_models['2'].save(f'temp.pt')
model_avg = YOLO('temp.pt')
metrics = model_avg.val(data='temp.yaml')
print(metrics.box.map)


metrics1 = server.val(data='temp.yaml')
print(metrics1.box.map)

# metrics2 = model1.val(data='temp_2.yaml')
# print(metrics2.box.map)