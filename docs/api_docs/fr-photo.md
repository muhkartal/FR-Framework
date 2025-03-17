# fr-photo API Reference

The `fr-photo` module provides functionality for detecting and recognizing faces in photos.

## Class: FRPhoto

The main class for photo-based face detection.

### Constructor

```python
FRPhoto(model_type='hog', detection_threshold=0.6, upsample_times=1)
```

#### Parameters:

-  `model_type` (str): Face detection model to use.
   -  `'hog'`: Faster CPU-based model (default)
   -  `'cnn'`: More accurate GPU-based model
-  `detection_threshold` (float): Confidence threshold for detection (0.0-1.0)
-  `upsample_times` (int): Number of times to upsample image (increases detection of small faces)

### Methods

#### detect_faces

```python
detect_faces(face_image, compute_encodings=True)
```

Detect faces in an image.

##### Parameters:

-  `face_image` (Union[FaceImage, np.ndarray, str]): Input image as:
   -  `FaceImage` object
   -  NumPy array
   -  Path to image file
-  `compute_encodings` (bool): Whether to compute face encodings (slower)

##### Returns:

-  List of `FaceDetection` objects

#### compare_faces

```python
compare_faces(known_encoding, face_encodings, tolerance=0.6)
```

Compare a known face encoding against a list of face encodings.

##### Parameters:

-  `known_encoding` (np.ndarray): Known face encoding
-  `face_encodings` (List[np.ndarray]): List of unknown face encodings to compare
-  `tolerance` (float): Tolerance for face comparison (lower is stricter)

##### Returns:

-  List of boolean values, True for matches

#### face_distance

```python
face_distance(face_encodings, face_to_compare)
```

Calculate face distance between encodings.

##### Parameters:

-  `face_encodings` (List[np.ndarray]): List of face encodings
-  `face_to_compare` (np.ndarray): Face encoding to compare against

##### Returns:

-  Array of distances (lower means more similar)

## Functions

### detect_faces

```python
detect_faces(image, **kwargs)
```

Convenience function to detect faces in an image.

#### Parameters:

-  `image` (Union[FaceImage, np.ndarray, str]): Input image
-  `**kwargs`: Additional arguments passed to `FRPhoto` constructor

#### Returns:

-  List of `FaceDetection` objects

## Example Usage

```python
from fr_framework.fr_photo import detect_faces

# Detect faces in an image
detections = detect_faces("path/to/image.jpg")

# Print results
for detection in detections:
    print(f"Face found at {detection.bbox} with confidence {detection.confidence}")

    # Access face encoding (if computed)
    if detection.encoding is not None:
        print(f"Encoding shape: {detection.encoding.shape}")
```
