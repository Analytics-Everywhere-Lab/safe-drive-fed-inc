import os
import shutil

# Define the path to the sorted_dataset folder
sorted_dataset_path = 'sorted_dataset'

# Iterate over each label folder (0, 1, ..., 5) in sorted_dataset
for label_folder in os.listdir(sorted_dataset_path):
    label_folder_path = os.path.join(sorted_dataset_path, label_folder)
    
    if os.path.isdir(label_folder_path):
        # Create 'images' and 'labels' subfolders inside each label folder
        images_subfolder = os.path.join(label_folder_path, 'images')
        labels_subfolder = os.path.join(label_folder_path, 'labels')
        
        if not os.path.exists(images_subfolder):
            os.makedirs(images_subfolder)
        
        if not os.path.exists(labels_subfolder):
            os.makedirs(labels_subfolder)
        
        # Iterate over files in the current label folder
        for file_name in os.listdir(label_folder_path):
            file_path = os.path.join(label_folder_path, file_name)
            
            # Skip the 'images' and 'labels' subfolders
            if os.path.isdir(file_path):
                continue
            
            # Move image files to the 'images' subfolder
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                shutil.move(file_path, os.path.join(images_subfolder, file_name))
            
            # Move label files to the 'labels' subfolder
            elif file_name.endswith('.txt'):
                shutil.move(file_path, os.path.join(labels_subfolder, file_name))

print("Files sorted into 'images' and 'labels' subfolders successfully.")
