import cv2
import os

# Ask user for their name
name = input("Enter your name: ").strip()

# Create folder for dataset if not exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Create folder for this person
user_path = f"dataset/{name}"
if not os.path.exists(user_path):
    os.makedirs(user_path)

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{user_path}/{str(count)}.jpg", face)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Face images saved!")
