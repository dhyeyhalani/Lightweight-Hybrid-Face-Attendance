# Hybrid Biometric Attendance System (HOG + SVM + MediaPipe)

A lightweight, CPU-optimized face recognition system that combines **Shape (HOG)**, **Texture (LBPH)**, and **3D Geometry (MediaPipe)** for high-accuracy attendance logging without using deep learning (Dlib/FaceNet).

## 🚀 Features
* **Hybrid Recognition Engine:** Fuses HOG features and MediaPipe 3D Landmarks using a Linear SVM.
* **Geometric Liveness Detection:** Prevents photo spoofing by tracking Eye Aspect Ratio (Blink Check) via 3D Mesh.
* **Zero-GPU Optimized:** Runs smoothly on standard CPUs (i3/i5) using a "Two-Speed Architecture" (Detection at 30 FPS, Recognition at 3 FPS).
* **Smart Subsampling:** Reduces training data size by 80% using Odd/Even frame selection.

## 🛠️ Tech Stack
* **OpenCV:** Image processing and LBPH.
* **MediaPipe:** 468-point 3D Face Mesh for geometric deep features.
* **Scikit-Learn:** SVM (Support Vector Machine) classifier.
* **Scikit-Image:** HOG (Histogram of Oriented Gradients) feature extraction.

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Lightweight-Face-Attendance.git](https://github.com/YOUR_USERNAME/Lightweight-Face-Attendance.git)
   cd Lightweight-Face-Attendance

2. Install dependencies:

Bash
pip install -r requirements.txt


🏃‍♂️ Usage
Run the main script:

Bash
python facialAttendance.py
Register a User: Press r to capture 10 seconds of data (automatically subsampled).

Mark Attendance: Stand in front of the camera. The system requires a Blink to verify liveness before logging.

Logs: Attendance is saved in Attendance.csv.

🧠 Architecture
The Guard (Fast Loop): Runs every 2 frames. Checks for blinks using MediaPipe.

The Detective (Slow Loop): Runs every 10 frames. Extracts HOG + Mesh features and classifies using SVM.


#### C. `.gitignore` (The Filter)
This file stops you from uploading junk. Create a file named `.gitignore` (no extension, just `.gitignore`) and paste this:

```text
# Ignore Virtual Environment
venv/
.venv/
env/

# Ignore PyCharm/VSCode settings
.idea/
.vscode/

# Ignore compiled python files
__pycache__/
*.pyc

# Ignore large raw video captures if you have any
*.mp4
*.avi

# (Optional) Ignore personal logs if you want a fresh start
Attendance.csv
