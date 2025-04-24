#CHatgpt told me :-

# 🪑 SeatScore: Real-Time Fatigue Detection & Seat Comfort Estimation

This project is a **modular real-time fatigue detection system** that computes a "SeatScore" using a combination of computer vision models and a trained regression decision tree.

It detects drowsiness using:
- YOLOv8 object detection (custom-trained)
- Eye Aspect Ratio (EAR) using Dlib facial landmarks
- Periodic DeepFace analysis for user demographics

It then predicts an overall comfort score ("SeatScore") using a trained regression model based on age, gender, and fatigue levels.

---

## 🚀 Features

- 🧠 **Deep Learning-Based Fatigue Detection** via YOLO
- 👀 **EAR (Eye Aspect Ratio)** based drowsiness estimation
- 📸 **DeepFace** for estimating age and gender
- 🌳 **DecisionTreeRegressor** for predicting comfort (SeatScore)
- 📊 Real-time visualization of fatigue levels and comfort score
- 📦 Pretrained model integration (`seatscore_tree.pkl`)

---

## 📁 File Structure

```bash
.
├── score.csv                # Training data for SeatScore model
├── seatscore_tree.pkl      # Trained regression tree model
├── shape_predictor_68_face_landmarks.dat  # Dlib model
├── yolov_awake_drowsy_11m_last.pt         # YOLO model weights
├── main.py                 # Main application loop (real-time detection)
├── seatscore.py            # SeatScore prediction utility
└── README.md               # This file
