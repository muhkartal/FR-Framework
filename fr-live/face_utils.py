
import os
import face_recognition
import cv2
import numpy as np
import logging
import pickle

def load_known_faces(data=None, directory=None, verbose=False):
    """
    Load known face encodings from given paths and IDs or from a directory.

    Args:
        data (dict): Dictionary with image paths and associated IDs.
        directory (str): Directory containing images of known faces.
        verbose (bool): If True, prints processing information for each face.

    Returns:
        list: A list of known face encodings.
        list: A list of known face IDs.
    """
    known_face_encodings = []
    known_face_ids = []

    # Option 1: Load faces from provided `data` dictionary.
    if data:
        for image_path, face_id in zip(data["images"], data["ids"]):
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image, num_jitters=10, model='large')

                if face_encodings:  # If at least one face is found
                    known_face_encodings.append(face_encodings[0])
                    known_face_ids.append(face_id)
                    if verbose:
                        print(f"Loaded encoding for {face_id} from {image_path}")
                else:
                    logging.warning(f"No faces found in image: {image_path}")
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")
    # Option 2: Load faces from a directory, where image file name is the ID.
    elif directory:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            face_id = os.path.splitext(filename)[0]  # Use file name as ID
            try:
                image = face_recognition.load_image_file(file_path)
                face_encodings = face_recognition.face_encodings(image, num_jitters=10, model='large')

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_ids.append(face_id)
                    if verbose:
                        print(f"Loaded encoding for {face_id} from {file_path}")
                else:
                    logging.warning(f"No faces found in image: {file_path}")
            except Exception as e:
                logging.error(f"Error processing image {file_path}: {e}")
    else:
        logging.error("Either 'data' or 'directory' must be provided to load known faces.")

    return known_face_encodings, known_face_ids



def save_new_face(face_id, face_encoding, encoding_file="face_encodings.pkl"):
    """
    Save a new face encoding and ID to a file.

    Args:
        face_id (str): The ID or name of the person.
        face_encoding (np.ndarray): The face encoding to save.
        encoding_file (str): Path to the file where encodings are saved.
    """
    if os.path.exists(encoding_file):
        with open(encoding_file, "rb") as file:
            data = pickle.load(file)
            known_face_ids = data["ids"]
            known_face_encodings = data["encodings"]
    else:
        known_face_ids = []
        known_face_encodings = []

    known_face_ids.append(face_id)
    known_face_encodings.append(face_encoding)

    with open(encoding_file, "wb") as file:
        pickle.dump({"ids": known_face_ids, "encodings": known_face_encodings}, file)

    print(f"Registered new face: {face_id}")



def draw_fancy_box(frame, top, right, bottom, left, color=(0, 255, 0), thickness=2, style='fancy', label=None):
    """
    Draw a fancy box around the detected face, with optional label.

    Args:
        frame (numpy.ndarray): The image frame to draw on.
        top (int): Top y-coordinate of the box.
        right (int): Right x-coordinate of the box.
        bottom (int): Bottom y-coordinate of the box.
        left (int): Left x-coordinate of the box.
        color (tuple): Color of the box (B, G, R).
        thickness (int): Thickness of the box lines.
        style (str): Style of the box ('fancy' or 'simple').
        label (str): Optional label text to display near the box.

    Returns:
        numpy.ndarray: The modified frame with drawn box and label.
    """
    if style == 'fancy':
        # Draw corner lines to create a fancy box
        length = 30  # Length of the decorative corners
        cv2.line(frame, (left, top), (left + length, top), color, thickness)
        cv2.line(frame, (left, top), (left, top + length), color, thickness)
        cv2.line(frame, (right, top), (right - length, top), color, thickness)
        cv2.line(frame, (right, top), (right, top + length), color, thickness)
        cv2.line(frame, (left, bottom), (left + length, bottom), color, thickness)
        cv2.line(frame, (left, bottom), (left, bottom - length), color, thickness)
        cv2.line(frame, (right, bottom), (right - length, bottom), color, thickness)
        cv2.line(frame, (right, bottom), (right, bottom - length), color, thickness)
    else:
        # Draw a simple rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

    # Optional: Draw the label near the face box
    if label:
        label_position = (left, top - 15 if top - 15 > 15 else top + 15)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return frame
