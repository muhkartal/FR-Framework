"""
Facial landmark detection module.
"""
import cv2
import numpy as np
import dlib
from typing import List, Dict, Union, Optional, Tuple

from common import (
    FaceImage,
    FaceDetection,
    FaceLandmarks,
    bbox_to_rect
)


class FRLandmark:
    """
    Facial landmark detection using dlib.
    """

    def __init__(self, predictor_path: Optional[str] = None):
        """
        Initialize the landmark detector.

        Args:
            predictor_path: Path to dlib's shape predictor model file
                           (e.g., 'shape_predictor_68_face_landmarks.dat')
        """
        # Default predictor path (you'll need to download this file separately)
        if predictor_path is None:
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"

        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading shape predictor: {e}. "
                              f"Make sure the file exists at {predictor_path}")

    def detect_landmarks(self,
                        face_image: Union[FaceImage, np.ndarray],
                        detections: List[FaceDetection]) -> List[FaceLandmarks]:
        """
        Detect facial landmarks for detected faces.

        Args:
            face_image: FaceImage object or numpy array
            detections: List of FaceDetection objects

        Returns:
            List of FaceLandmarks objects
        """
        # Ensure we have a FaceImage
        if isinstance(face_image, np.ndarray):
            face_image = FaceImage(image=face_image)

        results = []
        for detection in detections:
            # Convert bbox to dlib rectangle
            rect = bbox_to_rect(detection.bbox)

            # Detect landmarks
            shape = self.predictor(face_image.image, rect)

            # Convert landmarks to numpy arrays
            points = {}

            # Left eye (points 36-41)
            left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            points['left_eye'] = left_eye_pts

            # Right eye (points 42-47)
            right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
            points['right_eye'] = right_eye_pts

            # Nose (points 27-35)
            nose_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(27, 36)])
            points['nose'] = nose_pts

            # Mouth (points 48-68)
            mouth_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(48, 68)])
            points['mouth'] = mouth_pts

            # Jaw (points 0-16)
            jaw_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 17)])
            points['jaw'] = jaw_pts

            # All points
            all_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
            points['all'] = all_pts

            # Create landmark result
            landmarks = FaceLandmarks(points=points, face_detection=detection)
            results.append(landmarks)

        return results

    def calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Calculate the eye aspect ratio (EAR).

        Args:
            eye_points: 6 points defining the eye

        Returns:
            Eye aspect ratio (smaller values indicate more closed eyes)
        """
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])

        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear


# Convenience functions
def detect_landmarks(face_image, detections, **kwargs):
    """Convenience function to detect landmarks."""
    landmark_detector = FRLandmark(**kwargs)
    return landmark_detector.detect_landmarks(face_image, detections)
