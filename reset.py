import yaml
import shutil

source_file_path = 'init.yaml'

server_dest_file_path = f'server/config/train.yaml'
client1_dest_file_path = f'client1/config/train.yaml'
client2_dest_file_path = f'client2/config/train.yaml'

# Overwrite the config file to init
with open(source_file_path, 'r') as file:
    source_data = yaml.safe_load(file)

source_data['names'] = source_data['names']
source_data['nc'] = source_data['nc']
source_data['test'] = source_data['test']
source_data['train'] = source_data['train']
source_data['val'] = source_data['val'] 

with open(server_dest_file_path, 'w') as file:
    yaml.safe_dump(source_data, file, default_flow_style=False)

with open(client1_dest_file_path, 'w') as file:
    yaml.safe_dump(source_data, file, default_flow_style=False)

with open(client2_dest_file_path, 'w') as file:
    yaml.safe_dump(source_data, file, default_flow_style=False)

shutil.rmtree('runs/')