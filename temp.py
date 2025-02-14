# Used to test random ideas.
import os
import cv2
import json

# Define paths
clients = ['client1', 'client2', 'client3', 'client4', 'server']
base_path = '/incremental_dataset/round_5'


# Function to draw bounding boxes
def draw_bounding_boxes(image, labels):
    height, width, _ = image.shape
    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.split())
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (50, 50, 255), 8)  # Thicker box
        cv2.putText(image, str(int(class_id)), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 3)
    return image

# Process each client/server folder
for client in clients:
    # client_path = os.path.join(client, base_path)
    client_path = f'{client}{base_path}'
    # Create output directory if it doesn't exist
    output_path = f'temp/{client}' 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(client_path):
        for filename in files:
            print(root)
            if filename.endswith('.jpg') or filename.endswith('.png'):
                
                # Load image
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                
                # Load corresponding label
                label_path = os.path.join(os.path.dirname(root), 'labels', filename.replace('.jpg', '.txt').replace('.png', '.txt'))
                with open(label_path, 'r') as f:
                    labels = f.readlines()
                
                # Draw bounding boxes
                image = draw_bounding_boxes(image, labels)
                
                # Save the image
                output_image_path = os.path.join(output_path, filename)
                cv2.imwrite(output_image_path, image)