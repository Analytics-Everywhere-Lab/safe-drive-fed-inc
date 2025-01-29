import os
import shutil

# Define the path to the sorted_dataset folder
sorted_dataset_path = 'sorted_dataset'
rounds = {
    'round_1': [0, 1],
    'round_2': [0, 1, 2],
    'round_3': [0, 1, 2, 3],
    'round_4': [0, 1, 2, 3, 4],
    'round_5': [0, 1, 2, 3, 4, 5]
}

# Create rounds directories and copy relevant subfolders
for round_name, subfolder_indices in rounds.items():
    round_path = os.path.join('.', round_name)
    os.makedirs(round_path, exist_ok=True)
    os.makedirs(os.path.join(round_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(round_path, 'labels'), exist_ok=True)

    for index in subfolder_indices:
        subfolder_name = str(index)
        subfolder_images_path = os.path.join(sorted_dataset_path, subfolder_name, 'images')
        subfolder_labels_path = os.path.join(sorted_dataset_path, subfolder_name, 'labels')

        # Copy images
        if os.path.exists(subfolder_images_path):
            for file_name in os.listdir(subfolder_images_path):
                src_file = os.path.join(subfolder_images_path, file_name)
                dest_file = os.path.join(round_path, 'images', file_name)
                shutil.copy(src_file, dest_file)

        # Copy labels
        if os.path.exists(subfolder_labels_path):
            for file_name in os.listdir(subfolder_labels_path):
                src_file = os.path.join(subfolder_labels_path, file_name)
                dest_file = os.path.join(round_path, 'labels', file_name)
                shutil.copy(src_file, dest_file)

print("Files have been organized into the specified rounds successfully.")

