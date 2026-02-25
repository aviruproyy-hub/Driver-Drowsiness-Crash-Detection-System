import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time

# 1. Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308] 

def calculate_EAR(eye_points):
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    h = dist.euclidean(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

def calculate_MAR(mouth_points):
    v = dist.euclidean(mouth_points[0], mouth_points[1])
    h = dist.euclidean(mouth_points[2], mouth_points[3])
    return v / h

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# Constants & Variables
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.5
FRAME_CHECK = 20    # Frames eyes must stay closed to trigger alert
EYE_COUNTER = 0
YAWN_COUNT = 0
YAWN_COOLDOWN = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
            
            # --- CALCULATIONS ---
            ear = (calculate_EAR(landmarks[LEFT_EYE]) + calculate_EAR(landmarks[RIGHT_EYE])) / 2.0
            mar = calculate_MAR(landmarks[MOUTH])

            # --- DROWSINESS LOGIC ---
            if ear < EAR_THRESHOLD:
                EYE_COUNTER += 1
            else:
                if EYE_COUNTER > 0:
                    EYE_COUNTER -= 1 # Slowly drain the bar instead of instant reset

            # --- YAWN LOGIC ---
            if mar > MAR_THRESHOLD:
                if not YAWN_COOLDOWN:
                    YAWN_COUNT += 1
                    YAWN_COOLDOWN = True
                cv2.putText(frame, "YAWN DETECTED", (w//2 - 100, h - 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                YAWN_COOLDOWN = False

            # --- PART 2: UI PROGRESS BAR ---
            bar_x, bar_y = 50, 60
            bar_width, bar_height = 200, 20
            
            # Calculate fill based on EYE_COUNTER
            fill_level = min(EYE_COUNTER / FRAME_CHECK, 1.0)
            fill_w = int(fill_level * bar_width)
            
            # Color logic: Green -> Orange -> Red
            color = (0, 255, 0)
            if fill_level > 0.5: color = (0, 165, 255)
            if fill_level >= 1.0: 
                color = (0, 0, 255)
                cv2.putText(frame, "DROWSINESS ALERT!", (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Draw the UI
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_height), color, -1)
            cv2.putText(frame, f"FATIGUE LEVEL: {int(fill_level*100)}%", (bar_x, bar_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Dashboard Info
            cv2.putText(frame, f"Yawns: {YAWN_COUNT}", (w - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (30, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Driver Monitoring System v2.0', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()