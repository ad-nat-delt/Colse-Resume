import cv2
import numpy as np

# Load the pre-trained Caffe model for face detection
prototxt_path = 'deploy.prototxt'
caffemodel_path = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Specify the path to the video file
video_path = 'assets/Times Square.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the video capture was successful
    if not ret:
        print("Error reading video file")
        break

    # Resize the frame to a specific width for better performance
    target_width = 300
    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_height = int(target_width / aspect_ratio)
    frame = cv2.resize(frame, (target_width, target_height))

    # Use the Caffe model for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (target_width, target_height), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Extract faces from the detections
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([target_width, target_height, target_width, target_height])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))

    # Sort faces based on their size (larger faces might be closer)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    # Blur only the people in the background
    for i in range(len(faces) - 1):
        (x1, y1, w1, h1) = faces[i]
        (x2, y2, w2, h2) = faces[i + 1]

        # Calculate average face size
        avg_face_size = sum([(w * h) for (_, _, w, h) in faces]) / len(faces) if faces else 0
        threshold = 1.5 * avg_face_size  # Adjust the multiplier as needed

        # Check if the face is in the background based on size and position
        if y1 + h1 < y2 and (w1 * h1 > threshold) and (w2 * h2 > threshold):
            margin = 20
            face_roi = frame[max(0, y1 - margin):min(frame.shape[0], y1 + h1 + margin), max(0, x1 - margin):min(frame.shape[1], x1 + w1 + margin)]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[max(0, y1 - margin):min(frame.shape[0], y1 + h1 + margin), max(0, x1 - margin):min(frame.shape[1], x1 + w1 + margin)] = blurred_face

    # Display the resulting frame
    cv2.imshow('Improved Face Blurring', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
