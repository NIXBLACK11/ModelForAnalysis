import cv2
import numpy as np
import os

# Load YOLO
net = cv2.dnn.readNet("largeFiles/yolov3.weights", "largeFiles/yolov3.cfg")
classes = []
with open("largeFiles/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# Convert numpy array to a list
unconnected_out_layers = net.getUnconnectedOutLayers().flatten().tolist()

# Create output layers
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

def obj_detect_yolo(screenshot, output_folder):
    img = screenshot
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize list to store detected object names
    detected_objects = []

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.append(classes[class_id])

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

    # Save the marked image
    img_name = os.path.basename(img_path)
    marked_img_path = os.path.join(output_folder, "marked_" + img_name)
    cv2.imwrite(marked_img_path, img)

    return detected_objects

def object_detect(screenshot, videoGenre):
    # Detect objects and save the image with marked objects
    detected_objects = obj_detect_yolo(screenshot, "savedScreenshots")



