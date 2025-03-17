"""
Face analysis module for extracting facial attributes.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple

from common import (
    FaceImage,
    FaceDetection,
    FaceLandmarks
)


class FRAnalyze:
    """
    Face analysis for extracting facial attributes.
    """

    def __init__(self):
        """Initialize the face analyzer."""
        pass

    def analyze_face(self,
                    face_image: Union[FaceImage, np.ndarray],
                    detection: FaceDetection,
                    landmarks: Optional[FaceLandmarks] = None) -> Dict[str, Any]:
        """
        Analyze a face to extract attributes (emotion, age, gender, etc.)

        Args:
            face_image: FaceImage object or numpy array
            detection: FaceDetection object
            landmarks: Optional FaceLandmarks object

        Returns:
            Dictionary of facial attributes
        """
        # Ensure we have a FaceImage
        if isinstance(face_image, np.ndarray):
            face_image = FaceImage(image=face_image)

        # Extract face region if not already done
        if detection.face_image is None:
            from common import extract_face_region
            face_region = extract_face_region(face_image.image, detection)
        else:
            face_region = detection.face_image

        # Initialize results
        results = {}

        # Analyze face blur
        blur_score = self._analyze_blur(face_region)
        results['blur_score'] = blur_score
        results['is_blurry'] = blur_score < 100  # Arbitrary threshold

        # Analyze brightness
        brightness = self._analyze_brightness(face_region)
        results['brightness'] = brightness

        # Analyze face orientation if landmarks are available
        if landmarks is not None:
            orientation = self._analyze_orientation(landmarks)
            results['head_pose'] = orientation

            # Eye aspect ratio (for blink detection)
            if 'left_eye' in landmarks.points and 'right_eye' in landmarks.points:
                left_ear = self._calculate_eye_aspect_ratio(landmarks.points['left_eye'])
                right_ear = self._calculate_eye_aspect_ratio(landmarks.points['right_eye'])
                avg_ear = (left_ear + right_ear) / 2.0

                results['left_eye_aspect_ratio'] = left_ear
                results['right_eye_aspect_ratio'] = right_ear
                results['avg_eye_aspect_ratio'] = avg_ear
                results['eyes_closed'] = avg_ear < 0.2  # Arbitrary threshold

        return results

    def _analyze_blur(self, face_image: np.ndarray) -> float:
        """
        Analyze the blurriness of a face image.

        Args:
            face_image: Face region as numpy array

        Returns:
            Blur score (higher is less blurry)
        """
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image

        # Calculate Laplacian variance
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _analyze_brightness(self, face_image: np.ndarray) -> float:
        """
        Analyze the brightness of a face image.

        Args:
            face_image: Face region as numpy array

        Returns:
            Brightness value (0-255)
        """
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image

        # Calculate average pixel value
        return gray.mean()

    def _analyze_orientation(self, landmarks: FaceLandmarks) -> Dict[str, float]:
        """
        Estimate head pose from landmarks.

        Args:
            landmarks: FaceLandmarks object

        Returns:
            Dictionary with estimated pose angles (yaw, pitch, roll)
        """
        # This is a simplified implementation
        # A real implementation would use 3D model fitting

        # Get all face points
        face_points = landmarks.points.get('all')
        if face_points is None:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

        # Use eye landmarks for roll estimation
        left_eye_center = landmarks.points['left_eye'].mean(axis=0)
        right_eye_center = landmarks.points['right_eye'].mean(axis=0)

        # Calculate eye angle (roll)
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        roll = np.degrees(np.arctan2(dy, dx))

        # Simplified yaw estimation based on nose position relative to face center
        nose_tip = landmarks.points['nose'][-1]  # Tip of the nose
        face_center_x = (landmarks.face_detection.left + landmarks.face_detection.right) / 2
        face_width = landmarks.face_detection.width

        # Normalize to [-1, 1] range
        normalized_nose_x = 2 * (nose_tip[0] - face_center_x) / face_width
        yaw = normalized_nose_x * 45  # Convert to degrees (rough approximation)

        # Simplified pitch estimation
        # Would need more complex 3D model for accurate pitch
        pitch = 0.0

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }

    def _calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Calculate the eye aspect ratio (EAR).

        Args:
            eye_points: 6 points defining the eye

        Returns:
            Eye aspect ratio (smaller values indicate more closed eyes)
        """
        # Ensure we have 6 points
        if len(eye_points) != 6:
            return 0.0

        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])

        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear


# Convenience functions
def analyze_face(face_image, detection, landmarks=None, **kwargs):
    """Convenience function to analyze a face."""
    analyzer = FRAnalyze(**kwargs)
    return analyzer.analyze_face(face_image, detection, landmarks)
