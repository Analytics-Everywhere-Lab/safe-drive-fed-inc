import numpy as np
import cv2
import csv

class_names = ["no_seatbelt", "seatbelt",]

warning_counters = {
    'Seat Belt': 0,
    'Inattentive': 0,
    'Drowsiness': 0,
}

counter = 0
frames_to_keep_warning = 30

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    global counter
    det_img = image.copy()    
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    
    detected_classes = []
    
    
    
    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        detected_classes.append(label)
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    warning_status = update_warning_status(detected_classes)

    frame_number = "frame"+str(counter)
    
    save_warning_status_to_csv(warning_status,frame_number)
    counter+=1
    distraction_level = get_distraction_level(warning_status)
    draw_warnings(det_img, warning_status)

    
    
    cv2.putText(det_img, distraction_level, (50,350), cv2.FONT_HERSHEY_SIMPLEX,  2, (0, 0, 255), 5, cv2.LINE_AA)
    return det_img


def draw_box( image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def augment_bbox_singleclass(yolo_singlebbox, classes):
    bbox = yolo_singlebbox.split()
    num_class = int(bbox[0])
    class_label = class_names[num_class]
    values_bbox = list(map(float, bbox[1:]))
    single_augment_bb = values_bbox + [class_label]
    return single_augment_bb


def augment_bbox_multiclass(yolo_multibbox, classes):
    augment_bb_list = []
    bboxes = yolo_multibbox.split('\n')
    for one_bbox in bboxes:
        # print(one_bbox)
        if one_bbox:
            list_aug_bbox = augment_bbox_singleclass(one_bbox, classes)
            augment_bb_list.append(list_aug_bbox)
    return augment_bb_list

def single_bbox_to_yolo(tbbox, classes):
    if tbbox:
        class_num = classes.index(tbbox[-1])
        bboxes = list(tbbox)[:-1]
        bboxes.insert(0, class_num)
    else:
        bboxes = []
    return bboxes

def multi_bbox_to_yolo(multi_transform_bbox, classes):
    single_label = [single_bbox_to_yolo(one_bbox, classes) for one_bbox in multi_transform_bbox]
    return single_label

def convert_yolo_labels_to_bbox(size,x,y,w,h):
    box = np.zeros(4)
    dw = 1./size[0]
    dh = 1./size[1]
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    box[0] = x-(w/2.0)
    box[1] = x+(w/2.0)
    box[2] = y-(h/2.0)
    box[3] = y+(h/2.0)

    return (box)


def detect_faces(image, face_cascade):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Define weights for each class
# class_weights = {
#     'drowsiness': 0.8,
#     'mobile': 1.0,
#     'inattentive': 0.7,
#     'drinking': 0.6,
#     'no_seatbelt': 0.5
# }

class_weights = {
    'Seat Belt': 0.7,
    'Inattentive': 0.9,
    'Drowsiness': 0.8,

}

# Define distraction levels
def get_distraction_level(status):
    total_score = sum(class_weights[class_name] for class_name, w_status in status.items() if w_status)
    
    if total_score < 0.5:
        return 'No Distraction'
    elif total_score < 1.0:
        return 'Moderate Distraction'
    else:
        return 'Severe Distraction'


def draw_warnings(frame, warnings):
    # Define positions for the warning indicators on the frame
    positions = {
        'Seat Belt': (550, 50),
        'Drowsiness': (550, 150),
        'Inattentive': (550, 250),
        # ...
    }
    
    # Define colors for indicators
    off_color = (100, 100, 100)  # Grey
    on_color = (0, 0, 255)       # Red

    
    for warning, position in positions.items():
        # Check if warning is active and set the color
        color = on_color if warnings[warning] else off_color
        
        # Draw the warning indicator as a circle
        cv2.circle(frame, position, 20, color, -1)  # -1 fills the circle
        
        # Put the warning text
        cv2.putText(frame, warning, (50, position[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    return frame


def update_warning_status(detected_classes):
    global warning_counters

    

# Define the classes corresponding to each warning
    warning_classes = {
        'Seat Belt': 'no_seatbelt',
        'Inattentive': ['drinking', 'mobile', 'inattentive'],
        'Drowsiness': 'drowsiness',
    }
    
    warnings_status = {'Seat Belt': False, 'Drowsiness': False, 'Inattentive': False}
    
    for warning, classes in warning_classes.items():
        if isinstance(classes, str):
            classes = [classes]  # Single class becomes a list for consistency
        
        # If any of the related classes are detected, reset the counter
        if any(cls in detected_classes for cls in classes):
            warning_counters[warning] = frames_to_keep_warning
        
        # Decrement counter if greater than zero
        if warning_counters[warning] > 0:
            warning_counters[warning] -= 1

        # Set warning status
        warnings_status[warning] = warning_counters[warning] > 0
    return warnings_status

def save_warning_status_to_csv(warnings_status, f_number):
    fieldnames = ['Driver ID'] + ['Frame number'] + list(warnings_status.keys())
    with open("../csv/warning_status.csv", 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header if the file is new
        if csvfile.tell() == 0:
            writer.writeheader()

        row_data = {'Frame number': f_number,
                    'Driver ID' : "001"}
        row_data.update(warnings_status)
        
        # Write the warning status to the CSV file
        
        writer.writerow(row_data)
    