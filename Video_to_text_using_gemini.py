import streamlit as st
import cv2
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI
# st.title('Video Content Summarizer')
# uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

def process_video(video_file):
    # Temporary save uploaded video
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.getvalue())

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_directory = 'Selected_frames'
    os.makedirs(output_directory, exist_ok=True)

    selected_frames = []
    previous_frame = None
    threshold = 0.40  # Change threshold according to need

    for frame_idx in tqdm(range(n_frames), desc="Processing Frames"):
        ret, img = cap.read()
        if not ret:
            break
        if previous_frame is not None:
            b, g, r = cv2.split(img)
            ssim_b, _ = ssim(previous_frame[0], b, full=True)
            ssim_g, _ = ssim(previous_frame[1], g, full=True)
            ssim_r, _ = ssim(previous_frame[2], r, full=True)
            similarity_index = (ssim_b + ssim_g + ssim_r) / 3
            if similarity_index < threshold:
                selected_frames.append(img)
                frame_filename = os.path.join(output_directory, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(frame_filename, img)
        previous_frame = cv2.split(img)

    cap.release()
    return output_directory

def generate_summary(output_directory):
    model = genai.GenerativeModel('gemini-pro-vision')
    model1 = genai.GenerativeModel('gemini-pro')
    descriptions = []

    for frame_filename in os.listdir(output_directory):
        frame_path = os.path.join(output_directory, frame_filename)
        with Image.open(frame_path) as img:
            response = model.generate_content(["Provide a short description about the image.", img])
            descriptions.append(response.text)

    summary_prompt = "Summarize the Content"
    combined_text = " ".join(descriptions)
    summary_response = model1.generate_content([summary_prompt, combined_text])
    return summary_response.text
def show():
    st.title('Video Content Summarizer')
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    

    if uploaded_file is not None:
        output_directory = process_video(uploaded_file)
        summary_text = generate_summary(output_directory)
        st.subheader("Video Summary")
        st.write(summary_text)
