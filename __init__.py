"""
FR-Framework: A comprehensive face recognition framework
"""

__version__ = '0.1.0'

# Import main components for easy access
from fr_framework.common import preprocess_image, FaceImage
from fr_framework.fr_photo import detect_faces
from fr_framework.fr_landmark import detect_landmarks
from fr_framework.fr_analyze import analyze_face
from fr_framework.fr_live import LiveFaceDetector
from fr_framework.fr_video import VideoProcessor

# Version information
__all__ = [
    'preprocess_image',
    'FaceImage',
    'detect_faces',
    'detect_landmarks',
    'analyze_face',
    'LiveFaceDetector',
    'VideoProcessor',
]
