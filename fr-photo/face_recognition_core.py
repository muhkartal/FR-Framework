import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image  # PIL is used to handle the uploaded images
from io import BytesIO

# Utility function to display images using Streamlit
def imshow(image, title="Image", size=10):
    """
    Display an image using Streamlit.

    Parameters:
    - image (numpy array): The image to be displayed.
    - title (str): Title of the image.
    - size (int): Figure size.
    """
    st.image(image, caption=title, use_column_width=True)

# Function to register a face and add it to known encodings
def register_face(image, name):
    """
    Register a new face into the known_face_encodings list.

    Parameters:
    - image (numpy array): Image in numpy array format.
    - name (str): Name of the person.

    Returns:
    - face_encoding (numpy array): Face encoding of the registered face.
    """
    face_encoding = face_recognition.face_encodings(image)[0]
    return face_encoding, name

# Function to recognize faces in an image
def recognize_faces(frame, known_face_encodings, known_face_names, tolerance=0.6):
    """
    Recognize faces in the provided frame.

    Parameters:
    - frame (numpy array): The image frame in which faces are to be recognized.
    - known_face_encodings (list): List of known face encodings.
    - known_face_names (list): List of names corresponding to the known encodings.
    - tolerance (float): Tolerance level for matching faces.

    Returns:
    - frame (numpy array): The frame with bounding boxes and labels.
    - face_names (list): Names of recognized faces.
    """
    # Convert BGR image (OpenCV) to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        confidence = 1 - face_distances[best_match_index]  # Higher confidence means closer match

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        print(f"Detected: {name} with confidence: {confidence:.2f}")

        # Draw bounding boxes and labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            # Draw label with name and confidence
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 40), font, 0.8, (255, 255, 255), 1)
            confidence_text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, confidence_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    return frame, face_names

# Streamlit app starts here
def main():
    st.title("Face Recognition System")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Register Face", "Recognize Face"])

    # Known faces data storage
    if "known_face_encodings" not in st.session_state:
        st.session_state["known_face_encodings"] = []
    if "known_face_names" not in st.session_state:
        st.session_state["known_face_names"] = []

    if page == "Register Face":
        st.header("Register a New Face")

        # Enter name and upload image for registration
        name = st.text_input("Enter the person's name")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None and name:
            # Load image and get encoding
            image = np.array(Image.open(uploaded_file))  # Convert uploaded file to numpy array
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                face_encoding, name = register_face(image, name)
                st.session_state["known_face_encodings"].append(face_encoding)
                st.session_state["known_face_names"].append(name)
                st.success(f"Successfully registered {name}!")

                # Display the uploaded image
                imshow(image, title=f"Registered: {name}")
            else:
                st.warning("No face detected in the image. Please upload a clear photo with a visible face.")

    elif page == "Recognize Face":
        st.header("Recognize Faces in an Image")

        # Upload an image for recognition
        uploaded_file = st.file_uploader("Upload an Image to Recognize", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Convert file to OpenCV format
            image = np.array(Image.open(uploaded_file))  # Convert uploaded file to numpy array
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Perform face recognition
            result_frame, detected_names = recognize_faces(
                frame,
                st.session_state["known_face_encodings"],
                st.session_state["known_face_names"],
                tolerance=0.6
            )

            # Display results
            if len(detected_names) > 0:
                st.success(f"Detected: {', '.join(detected_names)}")
                imshow(result_frame, title="Face Recognition Results")
            else:
                st.warning("No known faces detected in the image.")

if __name__ == "__main__":
    main()
