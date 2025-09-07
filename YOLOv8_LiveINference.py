import cv2
import numpy as np
from ultralytics import YOLO

#The MQTT part is removed from the code

# Load YOLOv8 model
model = YOLO("D:/YOLOV8/test/runs/detect/train3/weights/best.pt")
#model = YOLO("yolov8m.pt")

# Constants for distance estimation and angular calculation
KNOWN_WIDTH = 0.6  # Known width of the object in meters
FOCAL_LENGTH = 250  # Focal length for distance estimation

# Horizontal Field of View (HFOV) of the camera in degrees (Adjust based on your camera)
HFOV = 130.0  # Camera's horizontal field of view in degrees

# Tolerance value for alignment
TOLERANCE_PIXELS = 20  # Number of pixels to tolerate for "perpendicular" determination

def estimate_distance(known_width, focal_length, width_in_pixels):
    if width_in_pixels == 0:
        return float('inf')
    return (known_width * focal_length) / width_in_pixels

def calculate_angular_offset(box_center_x, frame_center_x, frame_width, hfov):
    # Calculate the pixel offset
    pixel_offset = box_center_x - frame_center_x
    
    # Calculate the angular offset based on pixel offset and camera's HFOV
    angular_offset = (pixel_offset / frame_width) * hfov
    return angular_offset

# Open the webcam (laptop camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate the center of the camera frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # Perform object detection with YOLOv8
    results = model(frame)
    
    # Parse results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confs = result.boxes.conf.cpu().numpy()  # Get confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Get class IDs
        #print(f"classes list: {classes}") #Check number of element in detected classes
        # if len(classes) > 0:
        #     print('Object detected')
        # else:
        #     print('not detected')
        for box, conf, cls in zip(boxes, confs, classes):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box around the object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Object width in pixels
            object_width_pixels = x2 - x1
            
            # Estimate distance
            distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, object_width_pixels)
            
            # Calculate the center of the bounding box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            
            # Draw centerline of the bounding box
            cv2.line(frame, (box_center_x, y1), (box_center_x, y2), (0, 0, 255), 2)
            
            # Draw centerline of the camera frame
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 0, 0), 2)
            
            # Calculate angular offset
            angular_offset = calculate_angular_offset(box_center_x, frame_center_x, frame_width, HFOV)
            
            # Compare the centerline of the bounding box with the center of the camera frame
            center_diff = abs(box_center_x - frame_center_x)
            
            if center_diff <= TOLERANCE_PIXELS:
                alignment_status = "Perpendicular"
                color = (0, 255, 0)  # Green if aligned
            else:
                alignment_status = "Not Perpendicular"
                color = (0, 0, 255)  # Red if not aligned

            # Put the class, confidence, distance, and alignment status text on the frame
            class_name = model.names[int(cls)]
            label = f"{class_name} {conf:.2f} Distance: {distance:.2f} m"
            cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display the angular offset and alignment status
            cv2.putText(frame, f"Angle: {angular_offset:.2f} degrees", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, alignment_status, (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            #Chekcing the existence of classes and object in the frame
            print(f"class found:{cls}")
            dock_found = False
            print(f"Number of classes: {len(classes)}")
            #Check for detection of dock
            if class_name == 'side' or 'door':
                dock_found = True
                print('dock found')
            
            #Check Side found
            if 2 in classes:
                print('side found') 
            #Check Door found
            if 1 in classes:
                print('door found')
            #Check box found
            if 0 in classes:
                print('box found')
    # Show the frame
    cv2.imshow('YOLOv8 Object Detection with Angular Offset', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
