# Quick Start Guide

This guide will help you get started with FR-Framework by walking through some basic examples.

## Prerequisites

Before you begin, make sure you have:

-  Python 3.7 or higher
-  Required dependencies (OpenCV, dlib, face_recognition, NumPy)
-  FR-Framework installed (see [Installation Guide](installation.md))

## 1. Detecting Faces in an Image

Let's start with the most basic task: detecting faces in an image.

```python
from fr_framework import detect_faces
import cv2

# Detect faces in an image
image_path = "path/to/your/image.jpg"
detections = detect_faces(image_path)

# Print the number of faces detected
print(f"Found {len(detections)} faces")

# Load the image for visualization
image = cv2.imread(image_path)

# Draw bounding boxes around detected faces
for detection in detections:
    top, right, bottom, left = detection.bbox

    # Draw a rectangle around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2. Adding Facial Landmarks

Now, let's enhance our example by detecting facial landmarks:

```python
from fr_framework import detect_faces, detect_landmarks
import cv2

# Detect faces
image_path = "path/to/your/image.jpg"
detections = detect_faces(image_path)

# Detect landmarks for each face
landmarks = detect_landmarks(image_path, detections)

# Load the image for visualization
image = cv2.imread(image_path)

# Draw bounding boxes and landmarks
for i, (detection, face_landmarks) in enumerate(zip(detections, landmarks)):
    top, right, bottom, left = detection.bbox

    # Draw a rectangle around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw landmarks
    for feature_name, points in face_landmarks.points.items():
        if feature_name == 'all':
            continue

        # Different colors for different features
        if feature_name == 'left_eye' or feature_name == 'right_eye':
            color = (255, 0, 0)  # Blue for eyes
        elif feature_name == 'nose':
            color = (0, 255, 0)  # Green for nose
        elif feature_name == 'mouth':
            color = (0, 0, 255)  # Red for mouth
        else:
            color = (255, 255, 0)  # Yellow for other features

        # Draw points
        for point in points:
            x, y = point.astype(int)
            cv2.circle(image, (x, y), 2, color, -1)

# Display the result
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 3. Analyzing Face Quality

Let's add face quality analysis to our example:

```python
from fr_framework import detect_faces, detect_landmarks, analyze_face
import cv2

# Detect faces
image_path = "path/to/your/image.jpg"
detections = detect_faces(image_path)

# Detect landmarks
landmarks = detect_landmarks(image_path, detections)

# Load the image for visualization
image = cv2.imread(image_path)

# Process each face
for i, (detection, face_landmarks) in enumerate(zip(detections, landmarks)):
    # Analyze the face
    analysis = analyze_face(image_path, detection, face_landmarks)

    # Display analysis results
    top, right, bottom, left = detection.bbox

    # Draw a rectangle around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Add analysis text
    blur_text = f"Blur: {analysis['blur_score']:.1f}"
    cv2.putText(image, blur_text, (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    bright_text = f"Brightness: {analysis['brightness']:.1f}"
    cv2.putText(image, bright_text, (left, bottom + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Add head pose info
    pose = analysis['head_pose']
    pose_text = f"Yaw: {pose['yaw']:.1f}, Roll: {pose['roll']:.1f}"
    cv2.putText(image, pose_text, (left, bottom + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the result
cv2.imshow("Face Analysis", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 4. Real-time Face Detection

Here's how to implement real-time face detection using a webcam:

```python
import cv2
from fr_framework import LiveFaceDetector

# Create a live detector
detector = LiveFaceDetector(
    width=640,
    height=480,
    detection_interval=0.1  # Detect faces every 0.1 seconds
)

# Start detection
detector.start()

try:
    while True:
        # Get the latest frame with detections
        frame = detector.get_latest_frame()

        if frame is not None:
            # Show the frame
            cv2.imshow('Live Face Detection', frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
finally:
    # Clean up
    detector.stop()
    cv2.destroyAllWindows()
```

## 5. Processing a Video File

Finally, let's look at how to process a video file:

```python
from fr_framework import process_video

# Process a video file
results = process_video(
    "path/to/your/video.mp4",
    save_frames=True,
    output_folder="output_frames"
)

# Print summary
print(f"Processed {results['frames_processed']} frames")
print(f"Found faces in {results['frames_with_faces']} frames")
print(f"Detected {results['total_faces_detected']} faces in total")

# Print time ranges where faces appear
print("\nFace Detection Timeline:")
for time_range in results['detection_summary']['time_ranges']:
    print(f"  {time_range['face_count']} faces from "
          f"{time_range['start_time']:.2f}s to {time_range['end_time']:.2f}s "
          f"(duration: {time_range['duration']:.2f}s)")
```

## Next Steps

Now that you've seen the basics, you might want to:

1. Learn about [more advanced features](advanced-features.md)
2. Explore the [API reference](../api/index.md) for detailed documentation
3. Check out the [examples](../examples/) for more inspiration
