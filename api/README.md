# FR-Framework API

This directory contains the REST API server for the FR-Framework.

## API Endpoints

The API provides the following endpoints:

-  `POST /detect` - Detect faces in an image
-  `POST /landmarks` - Detect facial landmarks in an image
-  `POST /analyze` - Analyze facial attributes in an image
-  `POST /video` - Process a video file (background task)
-  `GET /tasks/{task_id}` - Get status of a background task
-  `GET /output/{file_path}` - Get an output file

## API Usage Examples

### Detect Faces

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "model_type=hog" \
  -F "detection_threshold=0.6" \
  -F "compute_encodings=true" \
  -F "return_image=true"
```

### Detect Landmarks

```bash
curl -X POST "http://localhost:8000/landmarks" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "return_image=true"
```

### Analyze Face

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "detect_landmarks=true" \
  -F "return_image=true"
```

### Process Video

```bash
curl -X POST "http://localhost:8000/video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4" \
  -F "model_type=hog" \
  -F "detection_threshold=0.6" \
  -F "skip_frames=5" \
  -F "save_frames=true" \
  -F "return_faces=true"
```

### Get Task Status

```bash
curl -X GET "http://localhost:8000/tasks/{task_id}" \
  -H "accept: application/json"
```

## Running the API Server

### Using Python

```bash
cd api
python server.py
```

### Using Docker

```bash
docker-compose up fr-api
```

## API Documentation

Once the server is running, you can access the API documentation at:

-  Swagger UI: `http://localhost:8000/docs`
-  ReDoc: `http://localhost:8000/redoc`
