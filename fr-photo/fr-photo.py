import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image  # Use PIL to handle uploaded images
from face_recognition_core import register_face, recognize_faces, imshow  # Import from your renamed module

# Define the main function for the Streamlit interface
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
        name = st.text_input("Enter the person's name")

        # Upload an image
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None and name:
            # Load image and get encoding
            image = np.array(Image.open(uploaded_file))  # Convert uploaded file to numpy array
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                st.session_state["known_face_encodings"].append(face_encoding)
                st.session_state["known_face_names"].append(name)
                st.success(f"Successfully registered {name}!")

                # Display the uploaded image
                st.image(image, caption=f"Registered: {name}", use_column_width=True)
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
                st.image(result_frame, channels="BGR", caption="Face Recognition Results", use_column_width=True)
            else:
                st.warning("No known faces detected in the image.")

if __name__ == "__main__":
    main()
