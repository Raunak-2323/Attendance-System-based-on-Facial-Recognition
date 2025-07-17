import streamlit as st
import cv2
import numpy as np
from PIL import Image

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    st.info("📷 Webcam started. Press 's' to capture.")

    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Show the webcam image
        frame = cv2.flip(frame, 1)
        img = frame.copy()
        st.image(frame, channels="BGR")

        key = cv2.waitKey(1)
        if key == ord("s"):
            st.success("✅ Image captured.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return img

# Example usage in streamlit section:
if menu == "📸 Register Face":
    name = st.text_input("Enter your name")
    if st.button("Open Webcam & Capture Face"):
        img = capture_from_webcam()
        if img is not None:
            cv2.imwrite(f"dataset/{name}.jpg", img)
            st.success("Image saved successfully!")
