#  Face Recognition Attendance System using OpenCV

A simple yet powerful Python project that uses **OpenCV** to recognize faces from a webcam and automatically mark attendance in a CSV file. The system is divided into three parts and works fully offline.

---

## ✨ Features

- 🎯 Face detection using Haar Cascades  
- 🧠 Face recognition using LBPH (Local Binary Pattern Histogram)  
- 🎥 Real-time webcam input  
- 📝 Automatic CSV attendance logging  
- 🔌 Works offline without internet

---

##  How the Project Works

1. **Face Data Collection (`face_data_collector.py`)**  
   This script uses OpenCV’s Haar Cascade Classifier to detect faces from your webcam. Once a face is detected, it captures and saves 50 grayscale images of that person inside a dedicated folder (`dataset/YourName`).  
    Just type your name — the system does the rest.

2. **Model Training (`face_trainer.py`)**  
   The collected faces are used to train a **Local Binary Pattern Histogram (LBPH)** recognizer — a reliable and efficient algorithm for face recognition. It also generates a `labels.pickle` file that maps each numeric ID to a person's name, and the trained model is saved as `trainer.yml`.

3. **Real-Time Recognition & Attendance (`face_recognizer.py`)**  
   This script loads the trained model and continuously checks webcam frames for faces. When a face is recognized with high confidence, it logs the name and current time into an `attendance.csv` file. Already-marked names are skipped to avoid duplication.

---

##  Tech Stack Used

- **Python 3** – Core language  
- **OpenCV** – Face detection and recognition (Haar Cascades + LBPH)  
- **NumPy** – Image data manipulation  
- **Pandas** – Attendance CSV handling  
- **Pickle** – Saving label mappings  
- **CSV** – For storing attendance logs

---


