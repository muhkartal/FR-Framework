"""
Video face recognition module.
"""
import cv2
import numpy as np
import os
import time
from typing import List, Dict, Union, Optional, Tuple, Callable

from common import (
    FaceImage,
    FaceDetection
)
from fr_photo import FRPhoto


class VideoProcessor:
    """
    Process videos for face recognition.
    """

    def __init__(self,
                model_type: str = 'hog',
                detection_threshold: float = 0.6,
                skip_frames: int = 5,
                output_folder: Optional[str] = None):
        """
        Initialize the video processor.

        Args:
            model_type: 'hog' (faster, CPU) or 'cnn' (more accurate, GPU)
            detection_threshold: Confidence threshold for detection
            skip_frames: Number of frames to skip between detections
            output_folder: Folder to save processed frames and results
        """
        self.model_type = model_type
        self.detection_threshold = detection_threshold
        self.skip_frames = skip_frames
        self.output_folder = output_folder

        # Initialize face detector
        self.face_detector = FRPhoto(
            model_type=model_type,
            detection_threshold=detection_threshold
        )

        # Create output folder if specified
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def process_video(self,
                     video_path: str,
                     save_frames: bool = False,
                     return_faces: bool = False,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict:
        """
        Process a video file for face recognition.

        Args:
            video_path: Path to the video file
            save_frames: Whether to save frames with detected faces
            return_faces: Whether to return extracted face images
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with processing results
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Results storage
        all_detections = {}
        extracted_faces = []
        unique_face_count = 0
        total_faces_detected = 0

        # Process the video
        frame_idx = 0

        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame to improve performance
            if frame_idx % self.skip_frames == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create face image
                face_image = FaceImage(image=rgb_frame)

                # Detect faces
                detections = self.face_detector.detect_faces(face_image)

                # Store detections for this frame
                if detections:
                    all_detections[frame_idx] = detections
                    total_faces_detected += len(detections)

                    # Extract and store face images if requested
                    if return_faces:
                        for i, detection in enumerate(detections):
                            if detection.face_image is not None:
                                face_info = {
                                    'frame_idx': frame_idx,
                                    'face_idx': i,
                                    'bbox': detection.bbox,
                                    'confidence': detection.confidence,
                                    'image': detection.face_image
                                }
                                extracted_faces.append(face_info)
                                unique_face_count += 1

                    # Save frames with detections if requested
                    if save_frames and self.output_folder:
                        # Draw boxes on a copy of the frame
                        annotated_frame = frame.copy()

                        for detection in detections:
                            top, right, bottom, left = detection.bbox

                            # Draw a rectangle around the face
                            cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                            # Add a label
                            label = f"Face: {detection.confidence:.2f}"
                            cv2.putText(annotated_frame, label, (left, top - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Save the annotated frame
                        output_path = os.path.join(self.output_folder, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(output_path, annotated_frame)

                # Update progress
                if progress_callback and frame_count > 0:
                    progress_callback(frame_idx, frame_count)

            # Increment frame index
            frame_idx += 1

        # Release the video capture
        cap.release()

        # Prepare results
        results = {
            'video_path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'resolution': (width, height),
            'frames_processed': frame_idx,
            'frames_with_faces': len(all_detections),
            'total_faces_detected': total_faces_detected,
            'unique_face_count': unique_face_count,
            'detection_summary': self._summarize_detections(all_detections, fps),
        }

        # Add extracted faces if requested
        if return_faces:
            results['extracted_faces'] = extracted_faces

        return results

    def _summarize_detections(self, all_detections: Dict[int, List[FaceDetection]], fps: float) -> Dict:
        """
        Summarize detection results.

        Args:
            all_detections: Dictionary mapping frame indices to face detections
            fps: Frames per second

        Returns:
            Summary dictionary
        """
        if not all_detections:
            return {
                'face_frequencies': [],
                'time_ranges': []
            }

        # Count faces per frame
        faces_per_frame = {idx: len(detections) for idx, detections in all_detections.items()}

        # Count frequency of each face count
        face_counts = list(faces_per_frame.values())
        unique_counts = sorted(set(face_counts))
        frequencies = {count: face_counts.count(count) for count in unique_counts}

        # Convert to list of dictionaries for easier use
        face_frequencies = [
            {'face_count': count, 'frequency': freq}
            for count, freq in frequencies.items()
        ]

        # Find time ranges for each detected face
        frame_indices = sorted(all_detections.keys())
        time_ranges = []

        if frame_indices:
            # Initialize with the first frame
            current_range = {
                'start_frame': frame_indices[0],
                'end_frame': frame_indices[0],
                'face_count': len(all_detections[frame_indices[0]])
            }

            # Process the rest of the frames
            for idx in frame_indices[1:]:
                face_count = len(all_detections[idx])

                # If sequential and same face count, extend the range
                if idx == current_range['end_frame'] + 1 and face_count == current_range['face_count']:
                    current_range['end_frame'] = idx
                else:
                    # Convert frames to time and add the range
                    current_range['start_time'] = current_range['start_frame'] / fps
                    current_range['end_time'] = current_range['end_frame'] / fps
                    current_range['duration'] = current_range['end_time'] - current_range['start_time']

                    time_ranges.append(current_range)

                    # Start a new range
                    current_range = {
                        'start_frame': idx,
                        'end_frame': idx,
                        'face_count': face_count
                    }

            # Add the last range
            current_range['start_time'] = current_range['start_frame'] / fps
            current_range['end_time'] = current_range['end_frame'] / fps
            current_range['duration'] = current_range['end_time'] - current_range['start_time']

            time_ranges.append(current_range)

        return {
            'face_frequencies': face_frequencies,
            'time_ranges': time_ranges
        }


# Convenience function
def process_video(video_path, **kwargs):
    """Process a video file for face recognition."""
    processor = VideoProcessor(**kwargs)
    return processor.process_video(video_path)
