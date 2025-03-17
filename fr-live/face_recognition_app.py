import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
from face_utils import load_known_faces, draw_fancy_box, save_new_face

# Load known faces (adjust paths as needed)
data = {
    "images": ["assets/sample_face.jpg"],  # Replace with actual paths
    "ids": ["Suspect"]
}

# Initialize page selection
page = st.sidebar.selectbox("Select a Page", options=["Face Recognition", "Face Registration"])

# -------------- Face Recognition Page --------------
if page == "Face Recognition":
    st.title("Face Recognition System")
    st.write("Welcome to the **Face Recognition System**. This page allows you to detect and recognize faces in real-time or from an uploaded video. "
             "To get started, choose the video source from the settings on the sidebar.")

    st.sidebar.subheader("Video Settings")

    # Video source and configuration options
    video_source = st.sidebar.selectbox("Select Video Source",
                                        options=["Webcam", "Upload Video"],
                                        help="Choose whether to use your webcam or upload a video file for face recognition.")
    show_fancy_box = st.sidebar.checkbox("Show Fancy Box", value=True,
                                         help="Enable this to show a decorative bounding box around detected faces.")
    frame_skip_rate = st.sidebar.slider("Frame Skip Rate", min_value=1, max_value=10, value=2, step=1,
                                        help="Adjust how often frames are processed. A lower number will increase accuracy but slow down the processing speed.")
    recognition_threshold = st.sidebar.slider("Recognition Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01,
                                              help="Set the threshold for face recognition confidence. Lower values increase sensitivity but might increase false positives.")

    # Load known face encodings when the page loads
    known_face_encodings, known_face_ids = load_known_faces(data=data)

    # If no known faces are loaded, notify the user
    if len(known_face_encodings) == 0:
        st.warning("No known face encodings loaded. Please register known faces in the 'Face Registration' section before starting recognition.")
    else:
        # Video processing section starts here
        uploaded_file = None
        if video_source == "Upload Video":
            uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"],
                                                     help="Upload a video file in .mp4, .avi, or .mov format for face recognition.")

        # Initialize session state for recognition
        if "start_recognition" not in st.session_state:
            st.session_state["start_recognition"] = False

        # Start face recognition
        start_button_clicked = st.button("Start Face Recognition", key="start_button")
        if start_button_clicked:
            st.session_state["start_recognition"] = True
            st.success("Face recognition started. Please wait for the system to detect faces.")

        # Stop face recognition
        if st.session_state["start_recognition"]:
            stop_button_clicked = st.sidebar.button("Stop Face Recognition", key="stop_button")
            if stop_button_clicked:
                st.session_state["start_recognition"] = False
                st.info("Face recognition has been stopped.")

            # Placeholder for video frames
            stframe = st.empty()

            # Initialize video capture based on source
            if video_source == "Webcam":
                video_capture = cv2.VideoCapture(0)
            elif uploaded_file is not None:
                video_capture = cv2.VideoCapture(uploaded_file.name)
            else:
                st.error("Please upload a video file to start recognition.")
                st.stop()

            frame_count = 0  # Initialize frame counter

            # Video processing loop
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames to improve performance
                if frame_count % frame_skip_rate != 0:
                    continue

                # Resize frame for faster processing and convert to RGB
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Detect faces and encode them
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Loop over detected faces and match with known faces
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=recognition_threshold)
                    face_id = "Unknown"
                    face_confidence = 0.0

                    # If a match is found, get the closest match
                    if any(matches):
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            face_id = known_face_ids[best_match_index]
                            face_confidence = 1 - face_distances[best_match_index]

                    # Draw fancy box and display face ID if enabled
                    if show_fancy_box:
                        frame = draw_fancy_box(frame, top * 4, right * 4, bottom * 4, left * 4)
                    cv2.putText(frame, f"{face_id} ({face_confidence:.2f})", (left * 4, top * 4 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the frame with face recognition details
                stframe.image(frame, channels="BGR")

                # Stop face recognition if the stop button is clicked
                if not st.session_state["start_recognition"]:
                    st.success("Face Recognition stopped.")
                    break

            # Release resources after stopping
            video_capture.release()
            cv2.destroyAllWindows()


elif page == "Face Registration":
    # -------------- Face Registration Page --------------
    st.title("Face Registration")
    st.write("Use this page to register new faces for recognition. You can upload an image, and the system will automatically detect the face. "
             "Once a face is detected, you can assign a name and save it for future recognition.")

    # File uploader to upload an image for face registration
    uploaded_image = st.file_uploader("Upload an image to register", type=["jpg", "jpeg", "png"],
                                      help="Upload an image in .jpg, .jpeg, or .png format. Ensure the image clearly shows the face.")

    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Convert image to RGB
        img_rgb = np.array(img.convert("RGB"))

        # Detect faces and encode them
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if len(face_encodings) == 0:
            st.warning("No faces detected in the uploaded image. Please try another image.")
        else:
            new_face_id = st.text_input("Enter the name for the face:",
                                        help="Enter a unique name for the face to be registered.")
            if st.button("Register Face"):
                if new_face_id:
                    # Save the new face encoding and ID (implement save_new_face in face_utils.py)
                    save_new_face(new_face_id, face_encodings[0])
                    st.success(f"Successfully registered face for {new_face_id}.")
                else:
                    st.error("Please enter a name for the face.")
