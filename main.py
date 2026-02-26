import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import winsound
import threading

# --- 1. Define Indices Globally (Cleanest Way) ---
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

def play_sound(frequency, duration):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

# --- Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# --- Constants & State ---
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.5
FRAME_CHECK = 20
EYE_COUNTER = 0
YAWN_COUNT = 0
YAWN_COOLDOWN = False
YAWN_ALERT_DONE = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # 1. CREATE UI OVERLAY (Header Bar)
    cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1) 
    cv2.line(frame, (0, 80), (w, 80), (0, 255, 0), 2)     

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
            
            # --- 2. Step-by-Step Calculations (Fixed Syntax) ---
            left_eye_pts = landmarks[LEFT_EYE]
            right_eye_pts = landmarks[RIGHT_EYE]
            mouth_pts = landmarks[MOUTH]

            ear = (calculate_EAR(left_eye_pts) + calculate_EAR(right_eye_pts)) / 2.0
            mar = calculate_MAR(mouth_pts)

            # --- Visual Mapping ---
            color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
            cv2.polylines(frame, [left_eye_pts], True, color, 1)
            cv2.polylines(frame, [right_eye_pts], True, color, 1)

            # --- Drowsiness/Yawn Logic ---
            if ear < EAR_THRESHOLD: 
                EYE_COUNTER += 1
            else: 
                if EYE_COUNTER > 0: EYE_COUNTER -= 1

            if mar > MAR_THRESHOLD:
                if not YAWN_COOLDOWN:
                    YAWN_COUNT += 1
                    YAWN_COOLDOWN = True
            else: 
                YAWN_COOLDOWN = False

            # --- SYSTEMATIC UI ---
            fill_level = min(EYE_COUNTER / FRAME_CHECK, 1.0)
            bar_w = int(fill_level * 150)
            bar_color = (0, 255, 0) if fill_level < 0.8 else (0, 0, 255)
            
            # cv2.putText(frame, "FATIGUE:", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.rectangle(frame, (20, 50), (230, 20), (50, 50, 50), -1) 
            cv2.rectangle(frame, (20, 50), (80 + bar_w, 20), bar_color, -1)
            
            status_text = "AWAKE"
            status_color = (0, 255, 0)
            if fill_level >= 1.0:
                status_text = "DROWSY!!"
                status_color = (0, 0, 255)
                if EYE_COUNTER == FRAME_CHECK: play_sound(1000, 3000)
            
            cv2.putText(frame, f"STATUS: {status_text}", (w // 2 - 80, 45), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)

            cv2.putText(frame, f"YAWNS: {YAWN_COUNT}", (w - 150, 45), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

            # Center Alert Pop-up
            if fill_level >= 1.0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2-220, h//2-40), (w//2+220, h//2+40), (0, 0, 150), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "WAKE UP! PULL OVER", (w//2-190, h//2+10), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)

            if YAWN_COUNT > 0 and YAWN_COUNT % 3 == 0:
                if not YAWN_ALERT_DONE:
                    play_sound(500, 5000)
                    YAWN_ALERT_DONE = True
                cv2.putText(frame, "3 YAWNS: TAKE A BREAK", (w//2-150, h-50), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            else: 
                YAWN_ALERT_DONE = False

    cv2.imshow('ADAS - Driver Monitor Pro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()