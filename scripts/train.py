'''
python3 train.py --pretrained_weigths yolo_pretrainedweights --config ../config/train.yaml --epochs 10 --image_size 640 --model yolov8m.pt --device 0
'''

import argparse
from ultralytics import YOLO



parser = argparse.ArgumentParser(description="Training yolo model")
parser.add_argument("--pretrained_weigths", default=False, help="path to pretrained weigths folder (weigths downloads automatically)")
parser.add_argument("--config", default=False, help="path to config file .yaml")
parser.add_argument("--epochs", default=False, type= int, help="number of epochs")
parser.add_argument("--image_size", default=False, help="image size")
parser.add_argument("--model", default='yolov8m.pt', help="name of yolo model")
parser.add_argument("--device", default=0,type=int, help="Run on CPU or GPU")
args = parser.parse_args()

pretrained_model = args.pretrained_weigths
config_file = args.config
epoch = args.epochs
im_size = args.image_size
model_name = args.model

model = YOLO('../'+pretrained_model+'/'+model_name) # download yolov8 pretrained weights to yolo_pretrainedweights directory


# Train the model
results = model.train(data=config_file, epochs=epoch, imgsz=im_size, device=args.device)