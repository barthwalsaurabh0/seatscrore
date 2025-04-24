
# SeatScore: A Dynamic Deep Learning Approach to Determine How Much You Deserve a Seat

**Course:** AIL721 - Deep Learning, IIT Delhi  
**Authors:** Kushagra Karar (2024CSY7554), Saurabh Barthwal (2024CSY7562), Vedant Sharma (2024CSY7553)

## Overview

SeatScore is an intelligent system that assigns a "seat deservingness" score to passengers in public transport based on fatigue, age, and gender.
## Project Structure

```
├── seatscore.py                     # module to use the trained regression tree model
├── seatscore_decision_tree_model_train.py   # Script for training the regression tree model
├── seatscore_infered_live.py       # Real-time inference and seat score visualization
├── shape_predictor_68_face_landmarks.dat    # Pre-trained facial landmark model (Dlib)
├── yolov_awake_drowsy_11m_last.pt  # Custom YOLOv11m model for fatigue detection
├── seatscore_tree.pkl              # Trained regression tree model
├── hyperparameters.yaml            # YOLO model training hyperparameters
├── results.csv                     # Final results and performance metrics
├── report.pdf                      # Detailed report (architecture, methodology, results)
└── README.md
```

## Requirements

- Python 3.8+
- Libraries:
  - `opencv-python`
  - `dlib`
  - `deepface`
  - `scikit-learn`
  - `torch`
  - `ultralytics` (for YOLOv11)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Decision Tree Model
```bash
python seatscore_decision_tree_model_train.py
```

### 2. Run Live Inference
```bash
python seatscore_infered_live.py
```

The script captures video feed, processes fatigue (via EAR and YOLO), estimates age & gender (via DeepFace), and overlays the calculated seat score.



For full methodology, data handling, and experiments, refer to `report.pdf`.

## License

For academic and educational use only.
