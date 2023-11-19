import cv2

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video stream using the default camera (change 0 to the appropriate camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Sort faces based on their y-coordinate (higher y-coordinate means closer to the camera)
    faces = sorted(faces, key=lambda f: f[1])

    # Blur only the people in the background
    for i in range(len(faces) - 1):
        (x1, y1, w1, h1) = faces[i]
        (x2, y2, w2, h2) = faces[i + 1]

        # Check if the face is in the background based on the y-coordinate
        if y1 + h1 < y2:
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
