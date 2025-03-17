#!/usr/bin/env python
"""
CLI demo for FR-Framework.

This script demonstrates the basic functionality of the framework
through a command-line interface.
"""
import os
import argparse
import cv2
import numpy as np
import time

# Import FR-Framework components
try:
    from fr_framework import (
        preprocess_image,
        detect_faces,
        detect_landmarks,
        analyze_face,
        process_video,
        LiveFaceDetector
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fr_framework.common import preprocess_image
    from fr_framework.fr_photo import detect_faces
    from fr_framework.fr_landmark import detect_landmarks
    from fr_framework.fr_analyze import analyze_face
    from fr_framework.fr_video import process_video
    from fr_framework.fr_live import LiveFaceDetector


def process_image(image_path, output_dir=None, show=True):
    """Process a single image."""
    print(f"Processing image: {image_path}")

    # Load and preprocess the image
    start_time = time.time()
    face_image = preprocess_image(image_path)
    preprocess_time = time.time() - start_time

    # Detect faces
    start_time = time.time()
    detections = detect_faces(face_image)
    detection_time = time.time() - start_time

    # Create a copy for visualization
    display_image = face_image.image.copy()
    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

    # Process each face
    print(f"Found {len(detections)} faces:")

    for i, detection in enumerate(detections):
        top, right, bottom, left = detection.bbox

        # Draw face rectangle
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Add face number
        cv2.putText(display_image, f"Face #{i+1}", (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"  Face #{i+1} at {detection.bbox}")

    # Detect landmarks
    if detections:
        start_time = time.time()
        try:
            landmarks = detect_landmarks(face_image, detections)
            landmark_time = time.time() - start_time

            # Draw landmarks
            for face_landmarks in landmarks:
                for feature_name, points in face_landmarks.points.items():
                    if feature_name == 'all':
                        continue

                    # Draw each point in the feature
                    color = (255, 0, 0)  # Blue for most features

                    if feature_name == 'left_eye':
                        color = (0, 255, 255)  # Yellow
                    elif feature_name == 'right_eye':
                        color = (0, 255, 255)  # Yellow
                    elif feature_name == 'nose':
                        color = (255, 0, 255)  # Magenta
                    elif feature_name == 'mouth':
                        color = (0, 0, 255)    # Red

                    # Draw points
                    for point in points:
                        x, y = point.astype(int)
                        cv2.circle(display_image, (x, y), 1, color, -1)

            # Analyze faces
            for i, (detection, landmark) in enumerate(zip(detections, landmarks)):
                analysis = analyze_face(face_image, detection, landmark)

                # Print analysis results
                print(f"  Face #{i+1} analysis:")
                print(f"    Blur score: {analysis['blur_score']:.2f}")
                print(f"    Is blurry: {analysis['is_blurry']}")
                print(f"    Brightness: {analysis['brightness']:.2f}")
                print(f"    Head pose: {analysis['head_pose']}")
                if 'eyes_closed' in analysis:
                    print(f"    Eyes closed: {analysis['eyes_closed']}")

                # Add head pose info to image
                pose = analysis['head_pose']
                pose_text = f"Yaw: {pose['yaw']:.1f}, Roll: {pose['roll']:.1f}"
                cv2.putText(display_image, pose_text, (left, bottom + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        except Exception as e:
            print(f"  Error detecting landmarks: {e}")
            landmark_time = 0
    else:
        landmark_time = 0

    # Print timing info
    print(f"Timing:")
    print(f"  Preprocessing: {preprocess_time:.3f}s")
    print(f"  Face detection: {detection_time:.3f}s")
    print(f"  Landmark detection: {landmark_time:.3f}s")
    print(f"  Total: {preprocess_time + detection_time + landmark_time:.3f}s")

    # Save or show the result
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, display_image)
        print(f"Saved result to: {output_path}")

    if show:
        cv2.imshow("Face Detection", display_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_file(video_path, output_dir=None, show=False):
    """Process a video file."""
    print(f"Processing video: {video_path}")

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define progress callback
    def progress_callback(current_frame, total_frames):
        if current_frame % 100 == 0 or current_frame == total_frames - 1:
            percent = (current_frame / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processing: {current_frame}/{total_frames} frames ({percent:.1f}%)")

    # Process the video
    start_time = time.time()
    results = process_video(
        video_path,
        save_frames=bool(output_dir),
        output_folder=output_dir,
        progress_callback=progress_callback
    )
    total_time = time.time() - start_time

    # Print results
    print("\nVideo Processing Results:")
    print(f"  Video: {results['video_path']}")
    print(f"  Resolution: {results['resolution'][0]}x{results['resolution'][1]}")
    print(f"  FPS: {results['fps']:.2f}")
    print(f"  Total frames: {results['frame_count']}")
    print(f"  Processed frames: {results['frames_processed']}")
    print(f"  Frames with faces: {results['frames_with_faces']}")
    print(f"  Total faces detected: {results['total_faces_detected']}")
    print(f"  Processing time: {total_time:.2f}s")

    # Face frequency
    print("\nFace Frequencies:")
    for freq in results['detection_summary']['face_frequencies']:
        print(f"  {freq['face_count']} face(s): {freq['frequency']} frames")

    # Time ranges
    print("\nDetection Ranges:")
    for time_range in results['detection_summary']['time_ranges']:
        print(f"  {time_range['face_count']} face(s) from "
              f"{time_range['start_time']:.2f}s to {time_range['end_time']:.2f}s "
              f"(duration: {time_range['duration']:.2f}s)")

    if output_dir:
        print(f"\nFrames with detections saved to: {output_dir}")


def live_detection():
    """Run live face detection from camera."""
    print("Starting live face detection...")
    print("Press ESC to exit")

    # Create and start the detector
    detector = LiveFaceDetector(
        width=640,
        height=480,
        fps=30,
        model_type='hog',
        detection_interval=0.1
    )

    detector.start()

    try:
        while True:
            # Get the latest frame with detections
            frame = detector.get_latest_frame()

            if frame is not None:
                # Get detection count
                detections = detector.get_latest_detections()

                # Add detection count
                cv2.putText(frame, f"Faces: {len(detections)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the frame
                cv2.imshow('Live Face Detection', frame)

            # Exit on ESC key
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
    finally:
        # Clean up
        detector.stop()
        cv2.destroyAllWindows()
        print("Live detection stopped")


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description='FR-Framework Demo')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Image processing command
    image_parser = subparsers.add_parser('image', help='Process an image')
    image_parser.add_argument('image_path', help='Path to the image file')
    image_parser.add_argument('-o', '--output', help='Output directory for results')
    image_parser.add_argument('--no-show', action='store_true', help='Do not display the result')

    # Video processing command
    video_parser = subparsers.add_parser('video', help='Process a video')
    video_parser.add_argument('video_path', help='Path to the video file')
    video_parser.add_argument('-o', '--output', help='Output directory for frames with detections')
    video_parser.add_argument('--show', action='store_true', help='Show the processed video')

    # Live detection command
    subparsers.add_parser('live', help='Run live face detection from camera')

    args = parser.parse_args()

    if args.command == 'image':
        process_image(args.image_path, args.output, not args.no_show)
    elif args.command == 'video':
        process_video_file(args.video_path, args.output, args.show)
    elif args.command == 'live':
        live_detection()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
