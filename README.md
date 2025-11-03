
# SeatScore: A Dynamic Deep Learning Approach to Determine How Much You Deserve a Seat

## Overview

SeatScore is an intelligent system that assigns a "seat deservingness" score to passengers in public transport based on fatigue, age, and gender.

---

## Abstract
In crowded public transport, deciding who deserves a seat is often left to personal judgment — leading to unfair situations.  
SeatScore introduces an intelligent, data-driven system that computes a seat-deservingness score for each passenger based on visual cues such as fatigue, age, and gender.  

By combining computer vision and deep learning, the system provides a fairer and more objective way to assign seats dynamically.

---

## Project Overview

### Objective
To build a smart system that assigns a seat score to each passenger based on:
- Fatigue level  
- Age  
- Gender

The final seat score is computed using a regression tree trained on normalized feature vectors derived from facial analysis.

---

## Methodology

### Inputs
- User Video Data

### Feature Extraction
| Feature | Model / Method | Description | Weight |
|----------|----------------|--------------|---------|
| Fatigue | Custom YOLOv11m (fine-tuned) + Eye Aspect Ratio (EAR) | Detects drowsiness and eye openness | 80% YOLO + 20% EAR |
| Age & Gender | DeepFace (VGG-Face backbone) | Estimates demographic features | – |

### Feature Normalization
All features are scaled to 0–100 and combined as:  
```
[ fatigue_norm, age_norm, gender_norm ]
```

### Label Generation
Ground truth seat scores are heuristically assigned using a review-based system:  
- Two team members label independently.  
- A third member reviews or averages the scores for fairness.  

Example synthetic data:  

| Fatigue | Age | Gender | Seat Score |
|----------|-----|---------|-------------|
| 10 | 60 | 1 (Male) | 76 |
| 85 | 40 | 1 | 75 |
| 20 | 35 | 2 (Female) | 44 |
| 87 | 57 | 1 | 90 |
| 60 | 22 | 2 | 63 |

---

## Model Architecture

### Fatigue Detection Pipeline
1. EAR (Eye Aspect Ratio) using Dlib’s 68-point facial landmark detector.  
2. Custom YOLOv11m model fine-tuned on awake/drowsy datasets.  
3. Weighted fatigue score:  
   ```
   Fatigue = 0.2 * EAR + 0.8 * YOLO
   ```

### Age & Gender Detection
Using DeepFace, based on VGG-Face, trained on IMDB-WIKI dataset.

### Regression Tree
- Model: DecisionTreeRegressor  
- Input: Normalized feature vectors  
- Output: Continuous Seat Score  
- Optimized via cross-validation (minimizing MSE)

---

## Results

### Fatigue Detection (YOLOv11m)
| Metric | Value |
|---------|--------|
| Precision | 0.916 |
| Recall | 0.931 |
| Dataset Size | 1796 images (10% test) |

Best augmentations:
- Mosaic: 1.0  
- Flip (L/R): 0.5  
- HSV Augmentation: (Hue=0.015, Saturation=0.7, Value=0.4)  
- Translation: 0.2  
- Scale: 0.6  

### Seat Score Prediction (Regression Tree)
| Metric | Value |
|---------|--------|
| MARD | 3.8 |
| MSE | 9.4 |
| Dataset | 100 (30 real + 70 synthetic) |

---

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


## Inference Pipeline

1. Fatigue Computation:  
   - YOLO-based drowsiness classification  
   - EAR tracking over 60s  
2. Face Analysis:  
   - DeepFace estimates age and gender  
3. Seat Score Prediction:  
   ```python
   predict_seatscore(age, gender, fatigue)
   ```
4. Visualization:  
   - Overlay YOLO fatigue, EAR fatigue, age, gender, and seat score on live video feed  

---

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


## Conclusion
SeatScore is a novel application of computer vision and deep learning to promote fairness in public seating systems.  
By combining fatigue, age, and gender features, the model produces an interpretable seat-deservingness score using a regression tree.  

Future work includes:
- Refining ground truth labeling  
- Incorporating real-world data at scale  
- Exploring ethical and social implications of automated fairness systems  

