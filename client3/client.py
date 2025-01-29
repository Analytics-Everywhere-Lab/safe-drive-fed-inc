# THIS CAN BE DELETED IF THE CURRENT FEDERATED LEARNING IS A GOOD IMPLEMENTATION

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
import os

from ultralytics import YOLO
from load_data import load_data_loaders

DEVICE = "cpu"

class Client(fl.client.NumPyClient):
    def __init__(
            self,
            model: YOLO,
            trainloader: DataLoader,
            testloader: DataLoader,
    ) -> None:
        self.model = YOLO("yolov8m.pt"),
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config: Dict[str, str]):
        return [param.detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]):
        self.set_parameters(parameters)
        self.model.train(data=config_file, epochs=epoch, imgsz=im_size)  # 
        # YOLO.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy = Model.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader.dataset), {"accuracy:": float(accuracy)}
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition_id", type=int, required=True, choices=range(0, 10))
    args = parser.parse_args()

    # Load data INCLUDE CUSTOM LOADING HERE
    cwd = os.getcwd()
    path = f"data/cifar10_split/subfolder_{args.partition_id}"
    path = os.path.join(cwd, path)
    
    trainloader, testloader = load_data_loaders(path)
    # Test data loading TO DISCARD
    for images, labels in trainloader:
        print(f'Images batch shape: {images.shape}')
        print(f'Labels batch shape: {labels.shape}')
        break

    # Load model INCLUDE CUSTOM MODEL LOADING
    model = Model.Net().to(DEVICE).train()

    client = Client(model, trainloader, testloader).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
