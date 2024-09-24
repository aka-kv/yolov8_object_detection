import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Initialize YOLO model
@st.cache_resource
def load_model():
    return YOLO('C:/Users/Keerthivasan/Desktop/InstaDataHelp Analytics/object_detection/yolov8n.pt')

model = load_model()

def detect_objects_in_image(image):
    results = model(image)
    return results[0].plot()

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use a common codec (H.264 / 'avc1')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # Create a temporary file to store the output video
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    frame_buffer = []
    buffer_size = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_buffer.append(frame)
        if len(frame_buffer) < buffer_size:
            continue

        accumulated_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
        frame_buffer.pop(0)

        # YOLO object detection on each frame
        results = model(accumulated_frame, conf=0.5, iou=0.5)
        
        detected_frame = results[0].plot()
        detected_frame_bgr = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)
        
        out.write(detected_frame_bgr)

    cap.release()
    out.release()

    return out_path

st.title("Object Detection with YOLOv8")

upload_type = st.radio("Select upload type:", ("Image", "Video"))

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            image = cv2.imread(tmp_file_path)
            if image is None:
                st.error("Error: Could not read the uploaded image.")
            else:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Detect Objects"):
                    result_image = detect_objects_in_image(image)
                    st.image(result_image, caption="Detection Result", use_column_width=True)
            
            # Clean up the temporary file
            os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.video(tmp_file_path)
            
            if st.button("Detect Objects"):
                with st.spinner("Processing video..."):
                    result_video_path = detect_objects_in_video(tmp_file_path)
                    
                if result_video_path:
                    # Display the processed video using st.video()
                    st.video(result_video_path)

                    # Optionally, provide a download link for the video
                    with open(result_video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name="detected_video.mp4",
                            mime="video/mp4"
                        )
                    
                    # Clean up the processed video file
                    os.unlink(result_video_path)
            
            # Clean up the temporary uploaded video file
            os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
