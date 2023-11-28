import cv2
import numpy as np

# Load the YOLO model from files at https://github.com/pjreddie/darknet/tree/master
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Create a VideoCapture object to capture video from the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Set the desired window width
window_width = 480

# Classes to analyze (0 corresponds to "person" and 41 corresponds to "umbrella" in COCO dataset)
classes_to_analyze = [["person", 0], ["cup", 41]]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Calculate the aspect ratio of the original frame
    aspect_ratio = width / height

    # Calculate the corresponding window height based on the desired width
    window_height = int(window_width / aspect_ratio)

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass through the network
    detections = net.forward(output_layer_names)

    # Post-process the detections with non-maximum suppression (NMS)
    boxes = []
    arrayOfConfidences = []

    for obj in detections[0]:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        for id_class in classes_to_analyze:
            if confidence > 0.5 and class_id == id_class[1]:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                arrayOfConfidences.append([id_class[0], float(confidence)])
    indexesSecondElements = [item[1] for item in arrayOfConfidences]
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, indexesSecondElements, 0.5, 0.4)

    # Draw rectangles for the remaining detections after NMS
    for i in range(len(indices)):
        index = int(indices[i])
        x, y, w, h = boxes[index]

        # Get the class name and confidence for the current object
        class_name = arrayOfConfidences[index][0]
        confidence = arrayOfConfidences[index][1]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize the frame while maintaining the aspect ratio
    frame = cv2.resize(frame, (window_width, window_height))

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
