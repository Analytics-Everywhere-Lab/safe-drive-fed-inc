import os

# Define the path to the main folder containing the 'images' and 'labels' subfolders
main_folder_path = 'round_2'
labels_folder_path = os.path.join(main_folder_path, 'labels')

# Function to filter labels in a given file
def filter_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Keep only lines that start with '0 ' or '1 '
    # filtered_lines = [line for line in lines if line.startswith('0 ') or line.startswith('3 ')]
    filtered_lines = [line for line in lines if line.startswith('1 ')]
    
    # Overwrite the file with filtered lines
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

# Iterate over each label file in the labels folder
if os.path.exists(labels_folder_path):
    for label_file in os.listdir(labels_folder_path):
        if label_file.endswith('.txt'):
            label_file_path = os.path.join(labels_folder_path, label_file)
            filter_labels(label_file_path)

print("Labels filtered successfully.")

