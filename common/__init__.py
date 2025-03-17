"""
Common utilities and data structures for the FR-Framework.
"""

from .face_utils import (
    preprocess_image,
    FaceImage,
    FaceDetection,
    FaceLandmarks,
    rect_to_bbox,
    bbox_to_rect
)

__all__ = [
    'preprocess_image',
    'FaceImage',
    'FaceDetection',
    'FaceLandmarks',
    'rect_to_bbox',
    'bbox_to_rect'
]
