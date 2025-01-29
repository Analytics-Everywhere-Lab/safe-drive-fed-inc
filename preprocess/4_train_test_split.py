import os
import shutil
import random

# Define the main directory containing round subfolders
main_folder_path = '.'
round_folders = [f'round_{i}' for i in range(1, 6)]

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Function to create subfolders
def create_subfolders(base_path):
    for split in ['train', 'val', 'test']:
        images_subfolder = os.path.join(base_path, split, 'images')
        labels_subfolder = os.path.join(base_path, split, 'labels')
        os.makedirs(images_subfolder, exist_ok=True)
        os.makedirs(labels_subfolder, exist_ok=True)

# Function to split files into train, val, and test
def split_files(files, base_path):
    random.shuffle(files)
    total_files = len(files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, split_files in splits.items():
        for file in split_files:
            image_src = os.path.join(base_path, 'images', file)
            label_src = os.path.join(base_path, 'labels', file.replace('.jpg', '.txt'))

            image_dest = os.path.join(base_path, split, 'images', file)
            label_dest = os.path.join(base_path, split, 'labels', file.replace('.jpg', '.txt'))

            shutil.move(image_src, image_dest)
            shutil.move(label_src, label_dest)

# Iterate over each round folder
for round_folder in round_folders:
    round_folder_path = os.path.join(main_folder_path, round_folder)
    if os.path.isdir(round_folder_path):
        create_subfolders(round_folder_path)
        
        # List all image files
        image_files = [f for f in os.listdir(os.path.join(round_folder_path, 'images')) if f.endswith('.jpg')]

        # Split and move files
        split_files(image_files, round_folder_path)
    # Delete the images and labels folder
    os.removedirs(os.path.join(round_folder_path, 'labels'))
    os.removedirs(os.path.join(round_folder_path, 'images'))

print("Files split into 'train', 'val', and 'test' subfolders successfully.")

# Remove the sorted dataset

