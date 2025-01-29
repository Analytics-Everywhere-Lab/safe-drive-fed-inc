import os

# Base directory containing the round folders
base_dir = '.'

# Function to filter labels in a given file based on allowed prefixes
def filter_labels(file_path, allowed_prefixes):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Keep only lines that start with allowed prefixes for the round
    filtered_lines = [line for line in lines if any(line.startswith(f'{prefix} ') for prefix in allowed_prefixes)]
    
    # Overwrite the file with filtered lines
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

# Dictionary mapping round numbers to their allowed prefixes
round_prefixes = {
    1: ['0', '1'],
    2: ['2'],
    3: ['3'],
    4: ['4'],
    5: ['5']
}

# Iterate over each round and process the labels
for round_number, prefixes in round_prefixes.items():
    round_labels_path = os.path.join(base_dir, f'round_{round_number}', 'labels')
    
    if os.path.exists(round_labels_path):  # Check if the directory exists
        print(f"Processing labels for round {round_number}...")
        
        for label_file in os.listdir(round_labels_path):
            if label_file.endswith('.txt'):
                label_file_path = os.path.join(round_labels_path, label_file)
                filter_labels(label_file_path, prefixes)
        
        print(f"Labels filtered successfully in round {round_number}.")
    else:
        print(f"Round {round_number} directory not found, skipping.")
