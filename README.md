![Open CV Github Frame](https://github.com/TH-Activities/saturday-hack-night-template/assets/90635335/78554b37-32b2-4488-a10c-5c68098d7776)

# Project Name

Depth-Based Face Blurring with OpenCV

This project demonstrates depth-based face blurring in real-time using OpenCV and the Haarcascades face detection. The script captures video frames from either a video file or a webcam, detects faces, and applies depth-based blurring to faces in the background.

## Requirements

- Python 3
- OpenCV

Install the required library using:

## Team members

1. [Adithya D](https://github.com/ad-nat-delt)
2. [Srutav Tarun](https://github.com/SrutavTarun)

## Link to product walkthrough

[link to video](Link Here)

## How it Works ?

This Python script uses the OpenCV library to perform real-time face blurring based on depth estimation. Here's a brief explanation of how the script works:

Import Libraries:

cv2: OpenCV library for computer vision tasks.
Load Haarcascades Face Detector:

The script uses the Haarcascades face detection classifier provided by OpenCV. This pre-trained model is capable of detecting faces in images.
Specify Video Source:

The script can either read frames from a video file specified by the video_path variable or capture frames from a webcam using cv2.VideoCapture(0).
Capture Frames:

In a loop, the script continuously captures frames from the video source.
Face Detection:

Convert the captured frame to grayscale for better face detection accuracy.
Use the Haarcascades face detector to identify faces in the grayscale frame.
Depth-Based Face Blurring:

Sort the detected faces based on their size (larger faces first, as they might be closer).
Iterate through the sorted faces and compare the size and position of adjacent faces.
If the size condition (w1 * h1 > 3000) and position condition (y1 + h1 < y2) are satisfied, it is assumed that the current face is in the background compared to the next one.
Blur the face region using Gaussian blur (cv2.GaussianBlur) to simulate depth-based blurring.
Display the Result:

Show the resulting frame with the applied depth-based face blurring using cv2.imshow.
Exit the Loop:

The loop continues until the user presses the 'q' key, at which point the script releases the video capture object and closes the window.
This script simulates a depth-based face blurring effect by comparing the sizes and positions of adjacent faces. Keep in mind that this is a simplified approach and may not accurately represent real-world depth information. For precise depth estimation, specialized depth-sensing cameras or additional information would be required.

## Demo Video

<video width="640" height="360" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Libraries used

OpenCV (cv2): OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides tools for image and video processing, including face detection and image blurring.

Installation: pip install opencv-python

```
pip install opencv-python
```

NumPy (np): NumPy is a powerful numerical computing library in Python. It is commonly used for working with arrays and matrices, making it useful for various mathematical operations.

Installation: pip install numpy

```
pip install numpy
```

## How to configure

No configurations required

## How to Run

In the terminal run 'python app.py'

```
python app.py
```