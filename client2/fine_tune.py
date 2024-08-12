from ultralytics import YOLO
import os 
import argparse

epochs = 50
image_size = 640

parser = argparse.ArgumentParser(description="Fine-tuning")
parser.add_argument("--if_fed", default='0')
args = parser.parse_args()

config_file = os.path.join(os.getcwd(), "client2/config/train.yaml")
# Override the global YOLO settings
os.environ["YOLO_CONFIG_DIR"] = config_file

model_path = ""
if args.if_fed == '0':
    model_path = os.path.join(os.getcwd(), "client2/saved_model/yolo-last.pt")
else:
    model_path = os.path.join(os.getcwd(), "client2/saved_model/yolo-avg.pt")

print(model_path)

model = YOLO(model_path)
# Freeze the previous layers here
# freeze = [f"model.{x}." for x in range(21)]  # freeze all but last layer
# for k, v in model.named_parameters():
#     v.requires_grad = True  # train all layers
#     if any(x in k for x in freeze):
#         # print(f"freezing {k}")
#         v.requires_grad = False

results = model.train(data=config_file, epochs=epochs, imgsz=image_size)

save_path = os.path.join(os.getcwd(), "client2/saved_model/yolo-last.pt")
model.ckpt = dict(model=model.model)
model.save(save_path)
print(f"Saved fine-tuned client2 model at: {save_path}")