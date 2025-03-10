import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Live Face Detection")

# Use WebRTC to access webcam
webrtc_streamer(key="face-detection", video_frame_callback=video_frame_callback)
