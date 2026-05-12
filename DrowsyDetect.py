import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import winsound
import threading
import time
from collections import deque

# --- Landmark Indices ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

# --- Thresholds ---
MAR_THRESHOLD = 0.5
FRAME_CHECK = 20

# Calibration
CALIBRATION_DURATION = 5        # seconds
EAR_CALIBRATION_FACTOR = 0.75   # threshold = baseline × this
EAR_MIN_THRESHOLD = 0.12        # absolute floor

# PERCLOS
PERCLOS_WINDOW_S = 30       # rolling window in seconds
PERCLOS_ALERT = 0.25        # >= 25% closed = fatigued

# HUD
PANEL_W = 210               # left panel width


def calculate_EAR(eye_points):
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    h = dist.euclidean(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_MAR(mouth_points):
    v = dist.euclidean(mouth_points[0], mouth_points[1])
    h = dist.euclidean(mouth_points[2], mouth_points[3])
    return v / h if h > 0 else 0.0


def play_sound(frequency, duration):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()


def draw_metric_bar(frame, label, value, max_val, x, y, w, color):
    """Draw a small labeled progress bar."""
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0.0
    filled = int(ratio * w)
    # Background track
    cv2.rectangle(frame, (x, y), (x + w, y + 10), (50, 50, 50), -1)
    # Filled portion
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + 10), color, -1)
    # Label above bar
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def draw_status_badge(frame, status, x, y, w):
    """Draw a colored status badge."""
    badge_colors = {
        "AWAKE":      (0, 160, 0),
        "DROWSY":     (0, 30, 220),
        "FATIGUED":   (0, 80, 200),
        "YAWNING":    (0, 170, 255),
        "NO FACE":    (80, 80, 80),
        "CALIBRATING":(0, 190, 190),
    }
    col = badge_colors.get(status, (100, 100, 100))
    cv2.rectangle(frame, (x, y), (x + w, y + 28), col, -1)
    fs = 0.55 if len(status) <= 8 else 0.45
    # Center text inside badge
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    tx = x + (w - tw) // 2
    cv2.putText(frame, status, (tx, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)


def draw_calib_progress(frame, fraction):
    """Draw a centered calibration progress bar."""
    fh, fw = frame.shape[:2]
    bar_x, bar_w, bar_h = fw // 4, fw // 2, 16
    bar_y = fh // 2 + 30
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    # Filled
    filled = int(fraction * bar_w)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      (0, 220, 220), -1)
    # Label
    pct = int(fraction * 100)
    cv2.putText(frame, f"Calibrating... {pct}%", (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 1)
    cv2.putText(frame, "Look at the camera with eyes open",
                (bar_x - 20, bar_y + bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_hud(frame, stats):
    """Draw a clean translucent left-side panel with key metrics."""
    fh, fw = frame.shape[:2]
    pw = PANEL_W

    # Translucent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, fh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Title + separator
    cv2.putText(frame, "DRIVER MONITOR", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)
    cv2.line(frame, (10, 28), (pw - 10, 28), (0, 200, 255), 1)

    # --- Metric bars ---
    ear = stats["ear"]
    perclos = stats["perclos"]
    ear_thresh = stats.get("ear_thresh", 0.22)
    ear_col = (0, 200, 0) if ear >= ear_thresh else (0, 50, 230)
    perc_col = (0, 200, 0) if perclos < PERCLOS_ALERT else (0, 50, 230)

    draw_metric_bar(frame, "EAR",     ear,     0.45, 10,  48, pw - 20, ear_col)
    draw_metric_bar(frame, "PERCLOS", perclos, 0.50, 10,  78, pw - 20, perc_col)

    # --- Key stats (compact) ---
    cv2.line(frame, (10, 98), (pw - 10, 98), (55, 55, 55), 1)

    items = [
        ("Yawns",   str(stats["yawns"])),
        ("Blinks",  str(stats["blinks"])),
        ("Session", stats["elapsed_str"]),
    ]
    y0 = 116
    for label, val in items:
        cv2.putText(frame, f"{label:<9}{val}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (185, 185, 185), 1)
        y0 += 20

    # --- Status badge at bottom ---
    draw_status_badge(frame, stats["status"], 10, fh - 42, pw - 20)


# --- Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(1)

# --- Calibration State ---
calib_values = []               # EAR samples during calibration
calib_start = time.time()
calibrated = False
EAR_THRESHOLD = None            # set dynamically after calibration

# --- State ---
EYE_COUNTER = 0
YAWN_COUNT = 0
YAWN_COOLDOWN = False
YAWN_ALERT_DONE = False

# PERCLOS state
perclos_deque = deque()     # (timestamp, is_closed) tuples
perclos_val = 0.0

# Blink state
total_blinks = 0
prev_eye_closed = False

# Session
session_start = time.time()

print(f"[INFO] Keep eyes open and face the camera — calibrating for {CALIBRATION_DURATION}s.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    ear = 0.0
    mar = 0.0
    status = "AWAKE"
    face_found = bool(results.multi_face_landmarks)

    if not face_found:
        status = "NO FACE"
        cv2.putText(frame, "No face detected",
                    (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    else:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([
                np.multiply([p.x, p.y], [w, h]).astype(int)
                for p in face_landmarks.landmark
            ])

            left_eye_pts = landmarks[LEFT_EYE]
            right_eye_pts = landmarks[RIGHT_EYE]
            mouth_pts = landmarks[MOUTH]

            ear = (calculate_EAR(left_eye_pts) + calculate_EAR(right_eye_pts)) / 2.0
            mar = calculate_MAR(mouth_pts)

            # ===================== CALIBRATION PHASE =====================
            if not calibrated:
                elapsed_c = now - calib_start
                progress = min(elapsed_c / CALIBRATION_DURATION, 1.0)

                if elapsed_c < CALIBRATION_DURATION:
                    calib_values.append(ear)
                    status = "CALIBRATING"
                    draw_calib_progress(frame, progress)
                else:
                    # Need enough samples for a reliable baseline
                    if len(calib_values) < 5:
                        calib_values = []
                        calib_start = now
                        print("[WARN] Not enough calibration data — restarting.")
                    else:
                        baseline_ear = float(np.mean(calib_values))
                        EAR_THRESHOLD = max(baseline_ear * EAR_CALIBRATION_FACTOR,
                                            EAR_MIN_THRESHOLD)
                        calibrated = True
                        print(f"[INFO] Calibration complete.  "
                              f"baseline={baseline_ear:.3f}  "
                              f"threshold={EAR_THRESHOLD:.3f}")

                # During calibration show HUD with placeholder values & skip detection
                draw_hud(frame, {
                    "ear": ear, "perclos": 0.0, "yawns": 0,
                    "blinks": 0, "elapsed_str": "00:00",
                    "status": status, "ear_thresh": 0.22,
                })
                cv2.imshow('ADAS - Driver Monitor Pro', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            # ============================================================

            is_closed = ear < EAR_THRESHOLD

            # --- PERCLOS ---
            perclos_deque.append((now, is_closed))
            # Prune entries older than the window
            while perclos_deque and (now - perclos_deque[0][0]) > PERCLOS_WINDOW_S:
                perclos_deque.popleft()
            if perclos_deque:
                perclos_val = sum(1 for _, c in perclos_deque if c) / len(perclos_deque)

            # --- Blink counter (simple: closed->open transition) ---
            if prev_eye_closed and not is_closed:
                total_blinks += 1
            prev_eye_closed = is_closed

            # --- Draw face contours ---
            eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
            mouth_color = (0, 255, 255) if mar < MAR_THRESHOLD else (0, 165, 255)
            cv2.polylines(frame, [left_eye_pts], True, eye_color, 1)
            cv2.polylines(frame, [right_eye_pts], True, eye_color, 1)
            cv2.polylines(frame, [mouth_pts], True, mouth_color, 1)

            # --- Drowsiness Logic ---
            if is_closed:
                EYE_COUNTER += 1
            else:
                if EYE_COUNTER > 0:
                    EYE_COUNTER -= 1

            # --- Yawn Logic ---
            if mar > MAR_THRESHOLD:
                if not YAWN_COOLDOWN:
                    YAWN_COUNT += 1
                    YAWN_COOLDOWN = True
            else:
                YAWN_COOLDOWN = False

            # --- Determine status ---
            fill_level = min(EYE_COUNTER / FRAME_CHECK, 1.0)

            if fill_level >= 1.0:
                status = "DROWSY"
                if EYE_COUNTER == FRAME_CHECK:
                    play_sound(1000, 3000)
                # Full-screen red flash
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (0, 0, 140), -1)
                cv2.addWeighted(flash, 0.18, frame, 0.82, 0, frame)
                # Center alert
                cv2.putText(frame, "WAKE UP! PULL OVER",
                            (w // 2 - 180, h // 2 + 10),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

            elif perclos_val >= PERCLOS_ALERT:
                status = "FATIGUED"

            elif mar > MAR_THRESHOLD:
                status = "YAWNING"

            # --- Yawn burst alert ---
            if YAWN_COUNT > 0 and YAWN_COUNT % 3 == 0:
                if not YAWN_ALERT_DONE:
                    play_sound(500, 5000)
                    YAWN_ALERT_DONE = True
                cv2.putText(frame, "3 YAWNS: TAKE A BREAK",
                            (w // 2 - 150, h - 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            else:
                YAWN_ALERT_DONE = False

    # --- Draw HUD ---
    elapsed = now - session_start
    elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))

    draw_hud(frame, {
        "ear": ear,
        "perclos": perclos_val,
        "yawns": YAWN_COUNT,
        "blinks": total_blinks,
        "elapsed_str": elapsed_str,
        "status": status,
        "ear_thresh": EAR_THRESHOLD or 0.22,
    })

    cv2.imshow('ADAS - Driver Monitor Pro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
