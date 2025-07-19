
#  Face Recognition Attendance System

A simple Python project that uses **OpenCV** to recognize faces from a webcam and automatically mark attendance in a CSV file. The system is divided into three parts:

---

##  Features

- Face detection using Haar Cascades  
- Face recognition using LBPH (Local Binary Pattern Histogram)  
- Real-time webcam input  
- Automatic CSV attendance logging  
- Works offline without internet

---

## ⚙️ Tech Stack Used

| Technology | Purpose |
|------------|---------|
| **Python 3** | Core programming language |
| **OpenCV** | Face detection & recognition |
| **NumPy** | Numerical operations on arrays |
| **Pandas** | Handling and writing CSV attendance data |
| **Haar Cascade (XML)** | For detecting faces in images |
| **LBPH Algorithm** | For face recognition training |
| **Pickle** | For saving and loading label mappings |
| **CSV File** | Simple local storage for attendance logs |

---

## 📁 Project Structure

```bash
📂 dataset/                # Folder containing captured face images
📄 face_data_collector.py  # Collects face images and stores them
📄 face_trainer.py         # Trains model and saves recognizer + label mapping
📄 face_recognizer.py      # Real-time recognition + marks attendance in CSV
📄 haarcascade_frontalface_default.xml  # Face detection model
📄 trainer.yml             # Saved trained recognizer
📄 labels.pickle           # Dictionary mapping user IDs to names
📄 attendance.csv          # Attendance log with name + timestamp
