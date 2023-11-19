import cv2
import numpy as np

# Load the pre-trained Caffe model for face detection
prototxt_path = 'deploy.prototxt'
caffemodel_path = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Specify the path to the video file
video_path = 'assets/meeting.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Set the desired window size
window_width = 1200
window_height = 800
cv2.namedWindow('Improved Face Blurring', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Improved Face Blurring', window_width, window_height)

# Keep track of blurred faces to avoid repeated blurring
blurred_faces = set()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the video capture was successful
    if not ret:
        # Set the frame position back to the beginning for looping
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Use the Caffe model for face detection on the original-sized frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (frame.shape[1], frame.shape[0]), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Extract faces from the detections
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
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
            # Check if the face has already been blurred to avoid repeated blurring
            if i not in blurred_faces:
                margin = 20
                face_roi = frame[max(0, y1 - margin):min(frame.shape[0], y1 + h1 + margin),
                                 max(0, x1 - margin):min(frame.shape[1], x1 + w1 + margin)]
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[max(0, y1 - margin):min(frame.shape[0], y1 + h1 + margin),
                      max(0, x1 - margin):min(frame.shape[1], x1 + w1 + margin)] = blurred_face

                # Mark the face as blurred to avoid repeating the process
                blurred_faces.add(i)

    # Resize the frame for display
    display_frame = cv2.resize(frame, (window_width, window_height))

    # Display the resulting frame
    cv2.imshow('Improved Face Blurring', display_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
