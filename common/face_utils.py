"""
Common face utilities and data structures.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict


@dataclass
class FaceImage:
    """Container for an image with face detection data."""
    image: np.ndarray
    path: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None

    def __post_init__(self):
        if self.height is None or self.width is None:
            self.height, self.width = self.image.shape[:2]


@dataclass
class FaceDetection:
    """Represents a detected face in an image."""
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float
    face_image: Optional[np.ndarray] = None
    encoding: Optional[np.ndarray] = None

    @property
    def top(self) -> int:
        return self.bbox[0]

    @property
    def right(self) -> int:
        return self.bbox[1]

    @property
    def bottom(self) -> int:
        return self.bbox[2]

    @property
    def left(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class FaceLandmarks:
    """Facial landmarks detection result."""
    points: Dict[str, np.ndarray]
    face_detection: FaceDetection

    @property
    def left_eye(self) -> np.ndarray:
        return self.points.get('left_eye')

    @property
    def right_eye(self) -> np.ndarray:
        return self.points.get('right_eye')

    @property
    def nose(self) -> np.ndarray:
        return self.points.get('nose')

    @property
    def mouth(self) -> np.ndarray:
        return self.points.get('mouth')

    @property
    def jaw(self) -> np.ndarray:
        return self.points.get('jaw')


def preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> FaceImage:
    """
    Load and preprocess an image for face recognition.

    Args:
        image_path: Path to the image file
        target_size: Optional tuple (width, height) to resize the image

    Returns:
        FaceImage object containing the preprocessed image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB (face_recognition uses RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if needed
    if target_size is not None:
        img = cv2.resize(img, target_size)

    return FaceImage(image=img, path=image_path)


def rect_to_bbox(rect) -> Tuple[int, int, int, int]:
    """
    Convert a dlib rectangle to a (top, right, bottom, left) tuple.

    Args:
        rect: A dlib rectangle object

    Returns:
        Tuple of (top, right, bottom, left)
    """
    return (rect.top(), rect.right(), rect.bottom(), rect.left())


def bbox_to_rect(bbox: Tuple[int, int, int, int]):
    """
    Convert a (top, right, bottom, left) tuple to a dlib rectangle.

    Args:
        bbox: Tuple of (top, right, bottom, left)

    Returns:
        A dlib rectangle object
    """
    try:
        import dlib
        top, right, bottom, left = bbox
        return dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
    except ImportError:
        raise ImportError("dlib is required to use bbox_to_rect")


def extract_face_region(image: np.ndarray, detection: FaceDetection,
                        padding: float = 0.0) -> np.ndarray:
    """
    Extract the face region from an image based on detection result.

    Args:
        image: Source image (numpy array)
        detection: FaceDetection object
        padding: Padding factor (0.1 = 10% padding on all sides)

    Returns:
        Face region as a numpy array
    """
    h, w = image.shape[:2]

    # Apply padding
    top, right, bottom, left = detection.bbox
    width = right - left
    height = bottom - top

    # Calculate padding values
    pad_w = int(width * padding)
    pad_h = int(height * padding)

    # Apply padding with bounds checking
    left = max(0, left - pad_w)
    top = max(0, top - pad_h)
    right = min(w, right + pad_w)
    bottom = min(h, bottom + pad_h)

    # Extract face region
    face_region = image[top:bottom, left:right]

    return face_region
