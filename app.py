import cv2

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the path to the video file
video_path = 'assets/Times Square.mp4'

# Open the video file
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Sort faces based on their size (larger faces might be closer)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    # Blur only the people in the background
    for i in range(len(faces) - 1):
        (x1, y1, w1, h1) = faces[i]
        (x2, y2, w2, h2) = faces[i + 1]

        # Check if the face is in the background based on size and position
        if y1 + h1 < y2 and w1 * h1 > 3000:  # Adjust the size threshold as needed
            face_roi = frame[y1:y1 + h1, x1:x1 + w1]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y1:y1 + h1, x1:x1 + w1] = blurred_face

    # Display the resulting frame
    cv2.imshow('Depth-based Face Blurring', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
