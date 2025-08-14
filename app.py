import streamlit as st
import cv2
import tempfile
import time
from inference.models.utils import get_roboflow_model

# === CONFIGURATION ===
MODEL_NAME = "amazon-accident-detection-o3juo"
MODEL_VERSION = "1"
API_KEY = "ktSFVMakkE69oahKbqtv"

# Load Roboflow model once
@st.cache_resource
def load_model():
    return get_roboflow_model(
        model_id=f"{MODEL_NAME}/{MODEL_VERSION}",
        api_key=API_KEY
    )

model = load_model()

st.title("🚗 Real-Time Accident Detection")
st.write("Upload a video and see bounding boxes live while it processes.")

# Upload video
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    # Open video
    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        st.stop()

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = 1 / fps if fps > 0 else 0.03  # fallback for missing FPS

    # Prepare output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Live display placeholders
    video_placeholder = st.empty()
    progress_bar = st.progress(0)

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)

        # Draw predictions
        for prediction in results[0].predictions:
            x_center = int(prediction.x)
            y_center = int(prediction.y)
            w = int(prediction.width)
            h = int(prediction.height)

            x0 = x_center - w // 2
            y0 = y_center - h // 2
            x1 = x_center + w // 2
            y1 = y_center + h // 2

            label = prediction.class_name
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(frame, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save frame to output
        out.write(frame)

        # Show frame in browser
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        frame_idx += 1
        progress_bar.progress(frame_idx / frame_count)

        # Sleep to match FPS for real-time feel
        time.sleep(frame_delay)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")

    # Show download button
    with open(temp_output, "rb") as f:
        st.download_button("Download processed video", f, file_name="processed_video.mp4")
