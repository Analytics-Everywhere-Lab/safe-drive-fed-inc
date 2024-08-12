'''
python3 train.py --pretrained_weights yolo_pretrainedweights --config ../config/train.yaml --epochs 10 --image_size 640 --model yolov8m.pt --device 0
'''

import argparse
from ultralytics import YOLO
import torch
import os


parser = argparse.ArgumentParser(description="Training yolo model")
# parser.add_argument("--pretrained_weights", default=False, help="path to pretrained weights folder (weights downloads automatically)")
parser.add_argument("--config", default=False, help="path to config file .yaml")
parser.add_argument("--epochs", default=False, type= int, help="number of epochs")
parser.add_argument("--image_size", default=False, help="image size")
parser.add_argument("--model", default='yolov8m.pt', help="name of yolo model")
parser.add_argument("--init", default="True", help="Initial round of training")
# parser.add_argument("--device", default=0, type=int, help="Run on CPU or GPU")
args = parser.parse_args()

cwd = os.getcwd() # this cwd is one level above server. Modified to include server/

config_file = os.path.join(cwd, "server/" + args.config)
# Override the global YOLO settings
os.environ["YOLO_CONFIG_DIR"] = config_file

epoch = args.epochs
im_size = args.image_size

# if args.init == True:
#     # pretrained_model = "yolo_pretrainedweights"
#     model_name = "yolov8m.pt"
#     # model = YOLO('../'+pretrained_model+'/'+model_name) # download yolov8 pretrained weights to yolo_pretrainedweights directory
#     model = YOLO(model_name)
#     model.info()
#     print("Init number of classes")
# else:
#     model = args.model
#     model.info()
#     print("Incremental classes")
model_name = ""
if args.init == "True":    
    model_name = "yolov8m.pt"
    print('Initializing training process')
else:
    model_name = os.path.join(os.getcwd(), "server/saved_model/" + args.model)
# model = YOLO('../'+pretrained_model+'/'+model_name) # download yolov8 pretrained weights to yolo_pretrainedweights directory
model = YOLO(model_name)

# Train the model
results = model.train(data=config_file, epochs=epoch, imgsz=im_size)

# Save model - ALL MODELS ARE SAVED AS yolo-last
save_path = os.path.join(os.getcwd(), "server/saved_model/yolo-last.pt")
model.ckpt = dict(model=model.model)
model.save(save_path)
print(f"Saved server model at: {save_path}")