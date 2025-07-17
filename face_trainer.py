import cv2
import numpy as np
import os

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Path to dataset
dataset_path = 'dataset'

# Store faces and labels
faces = []
labels = []
label_dict = {}
current_id = 0

# Loop over dataset folder
for name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, name)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_id] = name  # map numeric ID to name

    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Detect face
        face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            labels.append(current_id)

    current_id += 1

# Train the model
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

# Save label dictionary (name ↔ id) for future
import pickle
with open("labels.pickle", "wb") as f:
    pickle.dump(label_dict, f)

print("✅ Model trained and saved as 'trainer.yml'")
