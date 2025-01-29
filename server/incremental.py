from ultralytics import YOLO
import os
import yaml
import argparse

def parse_classes_list(arg_value):
    return arg_value.split(',')

parser = argparse.ArgumentParser(description="Incremental learning")
parser.add_argument("--round_name")
parser.add_argument("--classes_list", nargs="+")
args = parser.parse_args()

config_path = 'server/config/train.yaml'
epochs = 1 # replace with 100

yaml_file_path = 'server/config/train.yaml'
base_dataset_path = '../../incremental_dataset'
round_name = args.round_name
classes_list = args.classes_list

# MODIFY THE LINES TO DO THE FOLLOWING
# 1. Compare yolo-avg with yolo-last
model = ""
model_fed = YOLO('server/saved_model/yolo-avg.pt')
model_last = YOLO('server/saved_model/yolo-last.pt')

if model_fed is not None:       # Federated learning has occurred
    metrics_fed = model_fed.val(data='server/config/train.yaml')
    metrics_last = model_last.val(data='server/config/train.yaml')
    if metrics_fed.box.map > metrics_last.box.map:
        model = 'yolo-avg.pt'
    else:
        model = 'yolo-last.pt'

else:
    model = 'yolo-last.pt'

 # Update the config file. This is done after the testing so that the config from the previous round matches what the model was trained on
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)
data['names'] = classes_list
data['nc'] = len(classes_list)
data['train'] = os.path.join(base_dataset_path, round_name, 'train/images')
data['val'] = os.path.join(base_dataset_path, round_name, 'val/images')
data['test'] = os.path.join(base_dataset_path, round_name, 'test/images')

with open(yaml_file_path, 'w') as file:
    yaml.safe_dump(data, file, default_flow_style=False)


# 2. Freeze some layers of the model to preserve knowledge or progressively unfreeze - SKIP THIS (SERVER HAS ENOUGH COMPUTE)

# 3. Send selected model to train.py as an argument 
os.system(f"python3 server/scripts/train.py --config config/train.yaml --model {model} --epochs {epochs} --image_size 640 --init False")
