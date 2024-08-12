import argparse
import os
import random
import glob
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description="Train/Test/Val split ")
parser.add_argument("--image_data", default=False, help="path to training image dataset")
parser.add_argument("--label_data", default=False, help="path to training label dataset")
parser.add_argument("--output", default=False, help="path to output the train/test/val split")
args = parser.parse_args()

total_img = args.image_data
total_label = args.label_data

output = args.output

all_img_files = glob.glob(total_img+"/*")

#### Shuffle the list
random.shuffle(all_img_files)

# Split file names into train, val, and test sets
total_imgs = len(all_img_files)
train_end = int(total_imgs * 0.7) ## For training: 70% of all the data
val_end = train_end + int(total_imgs * 0.2) ## For validation: 20% of all the data


train_files_images = all_img_files[:train_end]
val_files_images = all_img_files[train_end:val_end]
test_files_images = all_img_files[val_end:]   # For testing: remaining 10%
 
train_files_labels = []
for img_file in train_files_images:
    base_name = Path(img_file).stem
    txt_file = base_name + '.txt'
    txt_file_path = total_label+'/'+txt_file
    train_files_labels.append(txt_file_path)

val_files_labels = []
for img_file in val_files_images:
    base_name = Path(img_file).stem
    txt_file = base_name + '.txt'
    txt_file_path = total_label+'/'+txt_file
    val_files_labels.append(txt_file_path)

test_files_labels = []
for img_file in test_files_images:
    base_name = Path(img_file).stem
    txt_file = base_name + '.txt'
    txt_file_path = total_label+'/'+txt_file
    test_files_labels.append(txt_file_path)


try:
    os.mkdir(output+"/training")

    os.mkdir(output+"/training/train")
    os.mkdir(output+"/training/train/images")
    os.mkdir(output+"/training/train/labels")

    os.mkdir(output+"/training/val")
    os.mkdir(output+"/training/val/images")
    os.mkdir(output+"/training/val/labels")

    os.mkdir(output+"/training/test")
    os.mkdir(output+"/training/test/images")
    os.mkdir(output+"/training/test/labels")


except OSError as error:
    print(error)

for img, label in zip(train_files_images, train_files_labels):
    shutil.copy(img,output+"/training/train/images")
    shutil.copy(label,output+"/training/train/labels")

for img, label in zip(val_files_images, val_files_labels):
    shutil.copy(img,output+"/training/val/images")
    shutil.copy(label,output+"/training/val/labels")

for img, label in zip(test_files_images, test_files_labels):
    shutil.copy(img,output+"/training/test/images")
    shutil.copy(label,output+"/training/test/labels")
