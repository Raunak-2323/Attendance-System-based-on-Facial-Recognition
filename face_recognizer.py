import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os

# Load recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load labels (id ↔ name)
with open("labels.pickle", 'rb') as f:
    label_dict = pickle.load(f)
    labels = {v: k for k, v in label_dict.items()}  # Flip id:name → name:id

# Setup CSV file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

# Track already marked names
already_marked = []

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < 70:  # Confidence threshold (lower is better)
            name = label_dict.get(id_, "Unknown")
            color = (0, 255, 0)

            if name not in already_marked:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df = pd.read_csv(attendance_file)
                df.loc[len(df.index)] = [name, now]
                df.to_csv(attendance_file, index=False)
                already_marked.append(name)

        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Attendance session ended. Data saved to 'attendance.csv'")
