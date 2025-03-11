import streamlit as st
import cv2
import numpy as np
from face_detector import detect_faces

st.title("Live Face Detection App 👀")

# Open the webcam
cap = cv2.VideoCapture(0)

frame_window = st.image([])

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    faces = detect_faces(frame)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert frame to RGB (Streamlit needs RGB format)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    frame_window.image(frame, channels="RGB")

cap.release()
