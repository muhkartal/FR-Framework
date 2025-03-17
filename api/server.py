"""
API server for FR-Framework.
"""
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import uuid
import os
import time
import shutil
import uvicorn
import logging
from io import BytesIO

# Import FR-Framework components
try:
    from fr_framework import (
        preprocess_image,
        detect_faces,
        detect_landmarks,
        analyze_face,
        process_video,
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fr_framework.common import preprocess_image, FaceImage
    from fr_framework.fr_photo import detect_faces
    from fr_framework.fr_landmark import detect_landmarks
    from fr_framework.fr_analyze import analyze_face
    from fr_framework.fr_video import process_video

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fr-api")

# Create FastAPI app
app = FastAPI(
    title="FR-Framework API",
    description="API for face detection, recognition, and analysis",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create temp directory for uploaded files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Create output directory for results
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
DEFAULT_MODEL_TYPE = "hog"  # 'hog' or 'cnn'
DEFAULT_DETECTION_THRESHOLD = 0.6

# Define Pydantic models for request/response
class DetectionRequest(BaseModel):
    model_type: str = DEFAULT_MODEL_TYPE
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    compute_encodings: bool = True
    return_image: bool = False

class LandmarkRequest(BaseModel):
    predictor_path: Optional[str] = None
    return_image: bool = False

class AnalysisRequest(BaseModel):
    detect_landmarks: bool = True
    return_image: bool = False

class VideoRequest(BaseModel):
    model_type: str = DEFAULT_MODEL_TYPE
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    skip_frames: int = 5
    save_frames: bool = False
    return_faces: bool = False

class BoundingBox(BaseModel):
    top: int
    right: int
    bottom: int
    left: int

class FaceDetectionResult(BaseModel):
    bbox: BoundingBox
    confidence: float
    encoding: Optional[List[float]] = None
    face_image_path: Optional[str] = None

class LandmarkPoint(BaseModel):
    x: int
    y: int

class FacialFeature(BaseModel):
    points: List[LandmarkPoint]

class FaceLandmarkResult(BaseModel):
    left_eye: Optional[FacialFeature] = None
    right_eye: Optional[FacialFeature] = None
    nose: Optional[FacialFeature] = None
    mouth: Optional[FacialFeature] = None
    jaw: Optional[FacialFeature] = None

class HeadPose(BaseModel):
    yaw: float
    pitch: float
    roll: float

class FaceAnalysisResult(BaseModel):
    blur_score: float
    is_blurry: bool
    brightness: float
    head_pose: Optional[HeadPose] = None
    left_eye_aspect_ratio: Optional[float] = None
    right_eye_aspect_ratio: Optional[float] = None
    avg_eye_aspect_ratio: Optional[float] = None
    eyes_closed: Optional[bool] = None

class TaskInfo(BaseModel):
    task_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result_url: Optional[str] = None

# Store for background tasks
task_store = {}

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "FR-Framework API",
        "version": "0.1.0",
        "endpoints": {
            "POST /detect": "Detect faces in an image",
            "POST /landmarks": "Detect facial landmarks",
            "POST /analyze": "Analyze face attributes",
            "POST /video": "Process a video file",
            "GET /tasks/{task_id}": "Get status of a background task",
        }
    }

@app.post("/detect")
async def detect_faces_endpoint(
    request: DetectionRequest = DetectionRequest(),
    file: UploadFile = File(...),
):
    """
    Detect faces in an uploaded image.
    """
    try:
        # Create a unique filename
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Save the uploaded file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect faces
        detections = detect_faces(
            filepath,
            model_type=request.model_type,
            detection_threshold=request.detection_threshold,
            compute_encodings=request.compute_encodings
        )

        # Prepare results
        results = []
        for i, detection in enumerate(detections):
            # Convert numpy array to list for JSON serialization
            encoding = None
            if detection.encoding is not None and request.compute_encodings:
                encoding = detection.encoding.tolist()

            # Save face image if requested
            face_image_path = None
            if request.return_image and detection.face_image is not None:
                face_filename = f"face_{uuid.uuid4()}.jpg"
                face_filepath = os.path.join(OUTPUT_DIR, face_filename)
                cv2.imwrite(face_filepath, cv2.cvtColor(detection.face_image, cv2.COLOR_RGB2BGR))
                face_image_path = f"/output/{face_filename}"

            # Create result
            result = FaceDetectionResult(
                bbox=BoundingBox(
                    top=detection.top,
                    right=detection.right,
                    bottom=detection.bottom,
                    left=detection.left
                ),
                confidence=float(detection.confidence),
                encoding=encoding,
                face_image_path=face_image_path
            )
            results.append(result)

        # Return results
        return {
            "file": file.filename,
            "num_faces": len(results),
            "faces": results
        }

    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.post("/landmarks")
async def detect_landmarks_endpoint(
    request: LandmarkRequest = LandmarkRequest(),
    file: UploadFile = File(...),
):
    """
    Detect facial landmarks in an uploaded image.
    """
    try:
        # Create a unique filename
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Save the uploaded file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect faces first
        detections = detect_faces(filepath)

        if not detections:
            return {
                "file": file.filename,
                "num_faces": 0,
                "message": "No faces detected in the image"
            }

        # Detect landmarks
        landmarks = detect_landmarks(
            filepath,
            detections,
            predictor_path=request.predictor_path
        )

        # Prepare results
        results = []
        for i, (detection, landmark) in enumerate(zip(detections, landmarks)):
            # Convert landmark points to expected format
            landmark_result = {}

            for feature_name, points in landmark.points.items():
                if feature_name == 'all':
                    continue

                landmark_result[feature_name] = FacialFeature(
                    points=[LandmarkPoint(x=int(p[0]), y=int(p[1])) for p in points]
                )

            # Create bbox result
            bbox = BoundingBox(
                top=detection.top,
                right=detection.right,
                bottom=detection.bottom,
                left=detection.left
            )

            # Add to results
            results.append({
                "bbox": bbox,
                "landmarks": landmark_result
            })

        # Return results
        return {
            "file": file.filename,
            "num_faces": len(results),
            "faces": results
        }

    except Exception as e:
        logger.error(f"Error in landmark detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.post("/analyze")
async def analyze_face_endpoint(
    request: AnalysisRequest = AnalysisRequest(),
    file: UploadFile = File(...),
):
    """
    Analyze facial attributes in an uploaded image.
    """
    try:
        # Create a unique filename
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Save the uploaded file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect faces first
        detections = detect_faces(filepath)

        if not detections:
            return {
                "file": file.filename,
                "num_faces": 0,
                "message": "No faces detected in the image"
            }

        # Detect landmarks if requested
        landmarks = None
        if request.detect_landmarks:
            landmarks = detect_landmarks(filepath, detections)

        # Analyze faces
        results = []
        for i, detection in enumerate(detections):
            # Get landmarks for this face if available
            face_landmarks = landmarks[i] if landmarks else None

            # Analyze face
            analysis = analyze_face(filepath, detection, face_landmarks)

            # Convert head pose to expected format
            head_pose = None
            if 'head_pose' in analysis:
                head_pose = HeadPose(**analysis['head_pose'])

            # Create analysis result
            analysis_result = FaceAnalysisResult(
                blur_score=analysis['blur_score'],
                is_blurry=analysis['is_blurry'],
                brightness=analysis['brightness'],
                head_pose=head_pose
            )

            # Add eye aspect ratio if available
            if 'left_eye_aspect_ratio' in analysis:
                analysis_result.left_eye_aspect_ratio = analysis['left_eye_aspect_ratio']
                analysis_result.right_eye_aspect_ratio = analysis['right_eye_aspect_ratio']
                analysis_result.avg_eye_aspect_ratio = analysis['avg_eye_aspect_ratio']
                analysis_result.eyes_closed = analysis['eyes_closed']

            # Create bbox result
            bbox = BoundingBox(
                top=detection.top,
                right=detection.right,
                bottom=detection.bottom,
                left=detection.left
            )

            # Add to results
            results.append({
                "bbox": bbox,
                "analysis": analysis_result
            })

        # Return results
        return {
            "file": file.filename,
            "num_faces": len(results),
            "faces": results
        }

    except Exception as e:
        logger.error(f"Error in face analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.post("/video")
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    request: VideoRequest = VideoRequest(),
    file: UploadFile = File(...),
):
    """
    Process a video file for face detection (long-running task).
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())

        # Create a unique filename
        filename = f"{task_id}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Save the uploaded file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create output directory for this task
        task_output_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(task_output_dir, exist_ok=True)

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            status="processing",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            completed_at=None,
            result_url=None
        )

        # Store task info
        task_store[task_id] = task_info

        # Define background task function
        def process_video_task(filepath, task_id, output_dir, request):
            try:
                # Process the video
                results = process_video(
                    filepath,
                    model_type=request.model_type,
                    detection_threshold=request.detection_threshold,
                    skip_frames=request.skip_frames,
                    save_frames=request.save_frames,
                    return_faces=request.return_faces,
                    output_folder=output_dir if request.save_frames else None
                )

                # Save results as JSON
                result_path = os.path.join(output_dir, "results.json")

                # Convert numpy arrays to lists for JSON serialization
                results_json = {}
                for key, value in results.items():
                    if key == 'extracted_faces':
                        # Handle extracted faces separately (save images)
                        faces_info = []
                        for i, face_info in enumerate(value):
                            # Save face image
                            face_filename = f"face_{i}.jpg"
                            face_path = os.path.join(output_dir, face_filename)
                            cv2.imwrite(face_path, cv2.cvtColor(face_info['image'], cv2.COLOR_RGB2BGR))

                            # Store face info without the image
                            face_data = {k: v for k, v in face_info.items() if k != 'image'}
                            face_data['image_path'] = f"/output/{task_id}/{face_filename}"
                            faces_info.append(face_data)

                        results_json[key] = faces_info
                    else:
                        # Handle other fields
                        results_json[key] = value

                # Write results to file
                import json
                with open(result_path, "w") as f:
                    json.dump(results_json, f, indent=2)

                # Update task status
                task_store[task_id].status = "completed"
                task_store[task_id].completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                task_store[task_id].result_url = f"/output/{task_id}/results.json"

                logger.info(f"Task {task_id} completed")

            except Exception as e:
                logger.error(f"Error processing video: {e}")
                task_store[task_id].status = "failed"
                task_store[task_id].completed_at = time.strftime("%Y-%m-%d %H:%M:%S")

            finally:
                # Clean up the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

        # Start the background task
        background_tasks.add_task(
            process_video_task,
            filepath,
            task_id,
            task_output_dir,
            request
        )

        # Return task info
        return {
            "message": "Video processing started",
            "task_id": task_id,
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a background task.
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_store[task_id]

@app.get("/output/{file_path:path}")
async def get_output_file(file_path: str):
    """
    Get an output file.
    """
    filepath = os.path.join(OUTPUT_DIR, file_path)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath)

# Run the API server
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
