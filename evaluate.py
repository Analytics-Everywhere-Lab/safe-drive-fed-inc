from ultralytics import YOLO
import os
import yaml

for round in range(1,6):
    server_model = YOLO()
    model_1 = YOLO()
    model_2 = YOLO()
    source_file_path = ''
    
    if round == 1:
        source_file_path = 'init.yaml'
        server_model = YOLO(f'runs/detect/init/weights/best.pt')
        model_1 = YOLO(f'runs/detect/init_fine_tune_client1/weights/best.pt')
        model_2 = YOLO(f'runs/detect/init_fine_tune_client2/weights/best.pt')
    else:
        source_file_path = f'round_{round}.yaml'
        server_model = YOLO(f'runs/detect/round_{round}/weights/best.pt')
        model_1 = YOLO(f'runs/detect/round_{round}_fine_tune_client1/weights/best.pt')
        model_2 = YOLO(f'runs/detect/round_{round}_fine_tune_client2/weights/best.pt')

    server_dest_file_path = f'server/config/train.yaml'
    client1_dest_file_path = f'client1/config/train.yaml'
    client2_dest_file_path = f'client2/config/train.yaml'
    
    # Overwrite the config file to match what was set during training
    with open(source_file_path, 'r') as file:
        source_data = yaml.safe_load(file)

    source_data['names'] = source_data['names']
    source_data['nc'] = source_data['nc']
    source_data['test'] = source_data['test']
    source_data['train'] = source_data['train']
    source_data['val'] = source_data['test'] # YOLOv8 recommends using this approach for testing

    with open(server_dest_file_path, 'w') as file:
        yaml.safe_dump(source_data, file, default_flow_style=False)

    with open(client1_dest_file_path, 'w') as file:
        yaml.safe_dump(source_data, file, default_flow_style=False)

    with open(client2_dest_file_path, 'w') as file:
        yaml.safe_dump(source_data, file, default_flow_style=False)

    server_metrics = server_model.val(save_json=True, data='server/config/train.yaml')  
    output_string = f'{server_metrics.results_dict}'
    output_file_path = 'runs/detect/val/metrics_output.txt'
    with open(output_file_path, 'w') as file:
        file.write(output_string)
    os.rename(f"runs/detect/val", f"runs/detect/test_{round}_server")

    model_1_metrics = model_1.val(save_json=True, data='client1/config/train.yaml')  
    output_string = f'{model_1_metrics.results_dict}'
    output_file_path = 'runs/detect/val/metrics_output.txt'
    with open(output_file_path, 'w') as file:
        file.write(output_string)
    os.rename(f"runs/detect/val", f"runs/detect/test_{round}_client1")

    model_2_metrics = model_2.val(save_json=True, data='client2/config/train.yaml')  
    output_string = f'{model_2_metrics.results_dict}'
    output_file_path = 'runs/detect/val/metrics_output.txt'
    with open(output_file_path, 'w') as file:
        file.write(output_string)
    os.rename(f"runs/detect/val", f"runs/detect/test_{round}_client2")