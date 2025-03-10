import streamlit as st
import cv2
import numpy as np
from face_detector import detect_faces

st.title("Live Face Detection App")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is default webcam

# Streamlit UI
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access the webcam.")
        break

    # Detect faces
    frame = detect_faces(frame)

    # Convert color from BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame
    stframe.image(frame, channels="RGB")

cap.release()

