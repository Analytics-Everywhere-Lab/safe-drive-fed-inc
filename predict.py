from ultralytics import YOLO
import cv2
import os

for round in range(1,6):
    model = YOLO()
    server_model = YOLO()
    model_1 = YOLO()
    model_2 = YOLO()
    source_file_path = ''
    
    if round == 1:
        server_model = YOLO(f'runs/detect/init/weights/best.pt')
        model_1 = YOLO(f'runs/detect/init_fed12345_fine_tune_client1/weights/best.pt')
        model_2 = YOLO(f'runs/detect/init_fed12345_fine_tune_client2/weights/best.pt')
    else:
        server_model = YOLO(f'runs/detect/round_{round}/weights/best.pt')
        model_1 = YOLO(f'runs/detect/round_{round}_fed12345_fine_tune_client1/weights/best.pt')
        model_2 = YOLO(f'runs/detect/round_{round}_fed12345_fine_tune_client2/weights/best.pt')

    devices = ['server', 'client1', 'client2']
    for device in devices:
        if device == 'server':
            model = server_model
        if device == 'client1':
            model = model_1
        if device == 'client2':
            model = model_2
        test_path = f'{device}/incremental_dataset/round_{round}/test/images'
        for filename in os.listdir(test_path):
            img_path = os.path.join(test_path, filename)
            results = model(img_path)
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                # result.show()  # display to screen
                # result.save(filename="server_round_5_result.jpg")
                os.makedirs(f'inference/{device}/round_{round}', exist_ok=True)
                save_path = f'inference/{device}/round_{round}/{filename}'
                print(save_path)
                result.save(filename=f'{save_path}')
            
# Re-place the config file for each round
# model = YOLO('runs/detect/round_5/weights/last.pt')
# model = YOLO('runs/detect/round_3_fed12345_fine_tune_client2/weights/last.pt')

# # Run batched inference on a list of images
# img1 = 'client2/incremental_dataset/round_3/test/images/Tb1_out101.jpg'
# # img = cv2.imread(img1)
# # img1 = cv2.resize(img, (1080, 1920))
# results = model(img1)  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     # result.show()  # display to screen
#     # result.save(filename="server_round_5_result.jpg")
#     result.save(filename="client2_round_3_result.jpg")