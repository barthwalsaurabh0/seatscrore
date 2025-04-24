import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from collections import deque
from deepface import DeepFace
import time
from seatscore import predict_seatscore

# ================== PARAMETERS ==================
YOLO_MODEL_PATH = 'last_l.pt'
LANDMARK_MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

FRAME_HISTORY_SEC = 10      # how many seconds to keep history
YOLO_WEIGHT = 0.7           # YOLO contribution to final score
EAR_WEIGHT = 0.3            # EAR contribution to final score
EAR_THRESHOLD = 0.24        # EAR fatigue threshold
YOLO_IGNORE_CLASSES = ['0', '1']

FACE_ANALYZE_INTERVAL = 10    # seconds between DeepFace analysis

awake_keywords = ['Awake']
drowsy_keywords = ['Drowsy']
# =================================================

# ================== HELPERS ==================
def eye_aspect_ratio(eye_points):
    eye = np.array([(point.x, point.y) for point in eye_points])
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def compute_percentage(history_deque):
    if history_deque:
        drowsy_frames = history_deque.count(1)
        return (drowsy_frames / len(history_deque)) * 100
    return 0.0

def update_history(deque_obj, value, max_len):
    deque_obj.append(value)
    if len(deque_obj) > max_len:
        deque_obj.popleft()

def analyze_face(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)

        if isinstance(analysis, list):
            analysis = analysis[0]  # take first face

        age = int(analysis['age'])
        gender_str = analysis['dominant_gender'].lower()  
        gender = 0 if gender_str == 'man' else 1  # m=0, f=1

        return age, gender

    except Exception as e:
        print("DeepFace error:", e)
        return None, None



# ================== INITIALIZATIONS ==================
model = YOLO(YOLO_MODEL_PATH)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
history_size = int(frame_rate * FRAME_HISTORY_SEC)

yolo_history = deque()
ear_history = deque()

age, gender = None, None
last_face_analysis_time = time.time()

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ### --- YOLO Inference --- ###
    results = model.predict(frame, stream=True)
    yolo_drowsy_this_frame = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            original_class = model.names.get(cls_id, 'Unknown')

            if original_class in YOLO_IGNORE_CLASSES:
                continue

            # Draw YOLO box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box
            cv2.putText(frame, original_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if original_class in drowsy_keywords:
                yolo_drowsy_this_frame = 1
                break  # one detection is enough

    update_history(yolo_history, yolo_drowsy_this_frame, history_size)

    ### --- EAR Inference --- ###
    faces = detector(gray)
    ear_drowsy_this_frame = 0

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [landmarks.part(i) for i in LEFT_EYE_IDX]
        right_eye = [landmarks.part(i) for i in RIGHT_EYE_IDX]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            ear_drowsy_this_frame = 1

        # Draw eye regions
        for eye_points, color in zip([left_eye, right_eye], [(0, 20, 255), (0, 20, 255)]):  # blue color
            pts = np.array([(p.x, p.y) for p in eye_points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        break  # one face is enough

    update_history(ear_history, ear_drowsy_this_frame, history_size)

    ### --- Face Analysis (every few seconds) --- ###
    if time.time() - last_face_analysis_time > FACE_ANALYZE_INTERVAL:
        temp_age, temp_gender = analyze_face(frame)
        if temp_age is not None and temp_gender is not None:
            age, gender = temp_age, temp_gender
        last_face_analysis_time = time.time()

    ### --- SCORE Computation --- ###
    yolo_fatigue = compute_percentage(yolo_history)
    ear_fatigue = compute_percentage(ear_history)

    final_fatigue = (YOLO_WEIGHT * yolo_fatigue) + (EAR_WEIGHT * ear_fatigue)

    seatscore = None
    if age is not None and gender is not None:
        seatscore = predict_seatscore(age, gender, final_fatigue)

    ### --- Display Info --- ###
    cv2.putText(frame, f"YOLO Fatigue: {yolo_fatigue:.2f}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, f"EAR Fatigue: {ear_fatigue:.2f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Final Fatigue: {final_fatigue:.2f}%", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if age is not None and gender is not None:
        gender_str = "M" if gender == 0 else "F"
        cv2.putText(frame, f"Age: {age}  Gender: {gender_str}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if seatscore is not None:
        cv2.putText(frame, f"Seat Score: {seatscore:.2f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    cv2.imshow('Full Modular Fatigue Detection + SeatScore', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
