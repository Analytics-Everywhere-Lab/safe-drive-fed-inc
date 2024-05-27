'''
python3 pseudolabel.py --model ../saved_model/best.pt --video ../IMG_3192.MOV --image_output ../output/images --label_output ../output/labels
'''

import argparse
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import time
import cv2


parser = argparse.ArgumentParser(description="Pseudo labeling script")
parser.add_argument("--model", default=False, help="path to trained model")
parser.add_argument("--video", default=False, help="path to input video")
parser.add_argument("--image_output", default=False, help="path to output image data")
parser.add_argument("--label_output", default=False, help="path to output labels data")
args = parser.parse_args()

img_out = args.image_output
label_out = args.label_output


model = YOLO(args.model)

def infer_on_video(video_path, model):
    # Open the video
    count = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame1 = frame.copy()
        
        
        if ret == True:
            
            with torch.no_grad():
                predictions = model.predict(frame, imgsz=640, conf=0.30)
    
    
                for pred in predictions:
                    annotator = Annotator(frame)
                    boxes = pred.boxes
                    for box in boxes:
                        b = box.xywhn  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        combined = torch.cat((c.unsqueeze(0).float(), b), dim=1)
                        data_list = combined.numpy().flatten().tolist()
                        data_list[0] = int(data_list[0])
                        data_str = ' '.join(map(str, data_list))
                        
                        count +=1
                        cv2.imwrite(img_out+'/'+str(count)+'.jpg', frame1)
                        with open(label_out+'/'+str(count)+'.txt', 'w') as f:
                            f.write(data_str)

                        annotator.box_label(box.xyxy[0], model.names[int(c)])
    
    
                img = annotator.result() 
    
            
            
            # Display the resulting frame
            cv2.imshow('Frame', img)
            
            # Press Q on keyboard to exit
            key = cv2.waitKey(25)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            if key == ord('n') or key == ord('p'):
                break
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)



infer_on_video(args.video, model)