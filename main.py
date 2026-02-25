import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# 1. Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308] # Inner lip landmarks

def calculate_EAR(eye_points):
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    h = dist.euclidean(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

def calculate_MAR(mouth_points):
    v = dist.euclidean(mouth_points[0], mouth_points[1]) # 13 to 14
    h = dist.euclidean(mouth_points[2], mouth_points[3]) # 78 to 308
    return v / h

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# Constants
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.6  # Adjust based on your test
EYE_COUNTER = 0
YAWN_COUNT = 0
YAWN_COOLDOWN = False # To prevent multiple counts for one yawn

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
            
            # 2. Calculations
            ear = (calculate_EAR(landmarks[LEFT_EYE]) + calculate_EAR(landmarks[RIGHT_EYE])) / 2.0
            mar = calculate_MAR(landmarks[MOUTH])

            # 3. Drowsiness (EAR) Logic
            if ear < EAR_THRESHOLD:
                EYE_COUNTER += 1
                if EYE_COUNTER >= 20:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                EYE_COUNTER = 0

            # 4. Yawn (MAR) Logic
            if mar > MAR_THRESHOLD:
                if not YAWN_COOLDOWN:
                    YAWN_COUNT += 1
                    YAWN_COOLDOWN = True
                cv2.putText(frame, "YAWNING DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                YAWN_COOLDOWN = False

            # Visual Debugging Info
            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Yawns: {YAWN_COUNT}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Advanced Driver Monitor', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()