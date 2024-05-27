'''
python3 augment.py --image_data ../dataset/annotated_data/images --label_data ../dataset/annotated_data/labels --image_output ../dataset/augmented_data/images --label_output ../dataset/augmented_data/labels

'''

import argparse
import random
import os
import glob
import cv2
from pathlib import Path
import albumentations as A
from utils import *


parser = argparse.ArgumentParser(description="Augmentation scripts")
parser.add_argument("--image_data", default=False, help="path to image dataset")
parser.add_argument("--label_data", default=False, help="path to label dataset")
parser.add_argument("--image_output", default=False, help="path to augmented image dataset")
parser.add_argument("--label_output", default=False, help="path to augmented labels dataset")
args = parser.parse_args()

our_classes = ["no_seatbelt", "seatbelt"]
img_path = args.image_data ## annotated image dir
label_path = args.label_data ## annotated label dir

augment_img_dir = args.image_output ## augmented image dir where output will store
augment_label_dir = args.label_output ## augmented label dir where output will store


imgs = glob.glob(img_path+'/*')
selected_files_labels = glob.glob(label_path+'/*')
print(len(selected_files_labels))
random.shuffle(selected_files_labels)

# 50% of training dataset
half_size = len(selected_files_labels) // 2    ## or int(len(data_list) * 0.20) for 20%
print(half_size)

### ISSUE : https://github.com/albumentations-team/albumentations/issues/459#issuecomment-734454278

# def replace_function(file_path, old_function_str, new_function_str):
#     with open(file_path, 'r') as file:
#         content = file.read()

#     if old_function_str not in content:
#         raise ValueError("Old function not found in the file.")

#     modified_content = content.replace(old_function_str, new_function_str)

#     with open(file_path, 'w') as file:
#         file.write(modified_content)

# # Example usage
# file_path = '/Users/aashishmamgain/Downloads/EdgeAI/edgeai_pr/lib/python3.12/site-packages/albumentations/core/bbox_utils.py'  # Path to Python file

# old_function_str = '''
# def check_bbox(bbox: BoxType) -> None:
#     """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
#     for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
#         if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
#             raise ValueError(f"Expected {name} for bbox {bbox} to be in the range [0.0, 1.0], got {value}.")
#     x_min, y_min, x_max, y_max = bbox[:4]
#     if x_max <= x_min:
#         raise ValueError(f"x_max is less than or equal to x_min for bbox {bbox}.")
#     if y_max <= y_min:
#         raise ValueError(f"y_max is less than or equal to y_min for bbox {bbox}.")
# '''

# new_function_str = '''
# def check_bbox(bbox: BoxType) -> None:
#     """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
#     #my added block
#     bbox=list(bbox)
#     for i in range(4):
#       if (bbox[i]<0) :
#         bbox[i]=0
#       elif (bbox[i]>1) :
#         bbox[i]=1
#     bbox=tuple(bbox)

#     for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
#         if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
#             raise ValueError(f"Expected {name} for bbox {bbox} to be in the range [0.0, 1.0], got {value}.")
#     x_min, y_min, x_max, y_max = bbox[:4]
#     if x_max <= x_min:
#         raise ValueError(f"x_max is less than or equal to x_min for bbox {bbox}.")
#     if y_max <= y_min:
#         raise ValueError(f"y_max is less than or equal to y_min for bbox {bbox}.")
# '''

# replace_function(file_path, old_function_str, new_function_str)
# '''

for yolo_str_labels in selected_files_labels[:half_size]:
    print(yolo_str_labels)
    img_base_name = Path(yolo_str_labels).stem
    img_file = img_base_name + '.jpg'
    img_file_path = img_path+'/'+img_file


    image = cv2.imread(img_file_path)
    labels_read = open(yolo_str_labels, "r").read()

    lines = [line.strip() for line in labels_read.split("\n") if line.strip()]
    # print(lines)
    album_bb_lists = augment_bbox_multiclass("\n".join(lines), our_classes) if len(lines) > 1 else [augment_bbox_singleclass("\n".join(lines), our_classes)]

    transform = A.Compose([
        # A.RandomCrop(width=300, height=300),
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        # A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
        # A.Resize(300, 300)
    ], bbox_params=A.BboxParams(format='yolo'))

    transformed = transform(image=image, bboxes=album_bb_lists)

    transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']

    transform_bbox_num = len(transformed_bboxes)

    if transform_bbox_num:
        trans_bboxes = multi_bbox_to_yolo(transformed_bboxes, our_classes) if transform_bbox_num > 1 else [single_bbox_to_yolo(transformed_bboxes[0], our_classes)]

    if not any(element < 0 for row in trans_bboxes for element in row):
        lab_out_pth = os.path.join(augment_label_dir, "aug_"+img_base_name+".txt")
        with open(lab_out_pth, 'w') as output:
            for bbox in trans_bboxes:
                updated_bbox = str(bbox).replace(',', '').replace('[', '').replace(']', '')
                output.write(updated_bbox + '\n')

    out_img_path = os.path.join(augment_img_dir, "aug_"+img_file)
    cv2.imwrite(out_img_path, transformed_image)

    print("Processed : Image : {}".format(out_img_path))