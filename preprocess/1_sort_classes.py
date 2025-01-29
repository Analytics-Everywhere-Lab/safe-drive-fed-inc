import os
import shutil

# Define the paths for the dataset, images, and labels
dataset_path = 'dataset'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Define the new directory to store the sorted files
output_dir = 'sorted_dataset'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all label files in the labels folder
for label_file in os.listdir(labels_path):
    if label_file.endswith('.txt'):
        label_file_path = os.path.join(labels_path, label_file)
        
        # Read the label file
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                label = line.split()[0]  # The label is the first element in the line
                
                # Create a folder for this label if it doesn't exist
                label_folder = os.path.join(output_dir, label, 'labels')
                image_folder = os.path.join(output_dir, label, 'images')
                #if not os.path.exists(label_folder):
                    #os.makedirs(label_folder)
                os.makedirs(label_folder, exist_ok=True)
                os.makedirs(image_folder, exist_ok=True)
                
                # Define the source paths for the label and image files
                source_label_path = label_file_path
                source_image_path = os.path.join(images_path, label_file.replace('.txt', '.jpg'))  # Assuming image format is .jpg
                print(source_image_path)
                
                # Define the destination paths
                dest_label_path = os.path.join(label_folder, label_file)
                dest_image_path = os.path.join(image_folder, label_file.replace('.txt', '.jpg'))
                print(dest_image_path)
                
                # Copy the label file to the destination folder
                shutil.copy(source_label_path, dest_label_path)
                shutil.copy(source_image_path, dest_image_path)
                
                # Copy the image file to the destination folder
                #if os.path.exists(source_image_path):
                 #   shutil.copy(source_image_path, dest_image_path)
                #else:
                 #   print(f"Image file {source_image_path} not found.")

print("Files sorted successfully.")

