import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time

# 1. EAR Calculation Function
def calculate_EAR(eye_points):
    # Vertical distances between landmarks
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    h = dist.euclidean(eye_points[0], eye_points[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# 2. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Indices for Left and Right Eyes in MediaPipe FaceMesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# 3. Parameters
EAR_THRESHOLD = 0.22  # Below this, eye is considered closed
EYE_CLOSED_SECONDS = 1.5 # Time in seconds to trigger alert
counter = 0
start_time = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            landmarks = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
            
            # Calculate EAR for both eyes
            left_ear = calculate_EAR(landmarks[LEFT_EYE])
            right_ear = calculate_EAR(landmarks[RIGHT_EYE])
            avg_ear = (left_ear + right_ear) / 2.0

            # 4. Drowsiness Detection Logic
            if avg_ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                
                elapsed_time = time.time() - start_time
                if elapsed_time >= EYE_CLOSED_SECONDS:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                start_time = None # Reset if eyes open

            # Display EAR for debugging
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Driver Monitor System', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()