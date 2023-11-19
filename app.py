import cv2
import numpy as np

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Capture the previous frame for frame differencing
    prev_frame = gray.copy()

    # Calculate frame difference
    frame_diff = cv2.absdiff(gray, prev_frame)

    # Detect faces in the frame difference
    faces_diff = face_cascade.detectMultiScale(frame_diff, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Check if faces_diff is non-empty before concatenating
    if faces_diff:
        # Combine the faces from Haarcascades and frame differencing
        faces = np.vstack((faces, faces_diff))


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
