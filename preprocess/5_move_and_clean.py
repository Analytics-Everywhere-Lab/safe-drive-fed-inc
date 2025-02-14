import os
import shutil
import random

# Define the main directory containing round subfolders
main_folder_path = '.'
round_folders = [f'round_{i}' for i in range(1, 6)]
# Move rounds to client folder
client_number = input("Enter the client number: ")
for round_folder in round_folders:
    round_folder_path = os.path.join(main_folder_path, round_folder)
    destination_path = os.path.join(f'../client{client_number}/incremental_dataset', round_folder)
    # destination_path = os.path.join(f'../server/incremental_dataset', round_folder)
    
    if os.path.isdir(round_folder_path):
        if os.path.exists(destination_path):
            # Merge contents if destination path exists
            for item in os.listdir(round_folder_path):
                s = os.path.join(round_folder_path, item)
                d = os.path.join(destination_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            shutil.rmtree(round_folder_path)
        else:
            shutil.move(round_folder_path, destination_path)
# Clear the data
shutil.rmtree('dataset/images')
shutil.rmtree('dataset/labels')
shutil.rmtree('sorted_dataset') 