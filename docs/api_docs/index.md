# API Reference

This section provides detailed documentation for each module and class in the FR-Framework.

## Modules

| Module                        | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| [fr-photo](fr-photo.md)       | Photo-based face detection and recognition        |
| [fr-landmark](fr-landmark.md) | Facial landmark detection                         |
| [fr-analyze](fr-analyze.md)   | Face analysis (blur detection, orientation, etc.) |
| [fr-live](fr-live.md)         | Real-time face recognition from camera feed       |
| [fr-video](fr-video.md)       | Video processing for face recognition             |
| [common](common.md)           | Common utilities and data structures              |

## Core Classes

### fr-photo

-  `FRPhoto`: Main class for photo-based face detection
-  `detect_faces()`: Convenience function for face detection

### fr-landmark

-  `FRLandmark`: Facial landmark detection using dlib
-  `detect_landmarks()`: Convenience function for landmark detection

### fr-analyze

-  `FRAnalyze`: Face analysis for extracting facial attributes
-  `analyze_face()`: Convenience function for face analysis

### fr-live

-  `LiveFaceDetector`: Real-time face recognition from camera feed
-  `create_live_detector()`: Convenience function to create a detector

### fr-video

-  `VideoProcessor`: Process videos for face recognition
-  `process_video()`: Convenience function for video processing

### common

-  `FaceImage`: Container for an image with face detection data
-  `FaceDetection`: Represents a detected face in an image
-  `FaceLandmarks`: Facial landmarks detection result
-  `preprocess_image()`: Load and preprocess an image for face recognition
