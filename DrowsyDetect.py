import cv2
import mediapipe as mp
import numpy as np
import time
import platform
import threading
import sys
from collections import deque
from datetime import datetime
from math import sqrt

#THRESHOLDS

# Camera
CAM_INDEX            = 1        # webcam index
CAM_WIDTH            = 640      # pixels — keep consistent for pixel-based thresholds
CAM_HEIGHT           = 480
CAM_FLIP             = True     # True = mirror (driver-facing cam)

# EAR / drowsiness
EAR_CALIBRATION_FACTOR  = 0.75  # threshold = baseline × this
DROWSY_TIME_THRESHOLD   = 1.5   # seconds eyes must stay closed → DROWSY
EAR_SMOOTH_LEN          = 10    # rolling mean window (frames)

# Blink detection
BLINK_REOPEN_MAX_S   = 0.5      # eye closure ≤ this = blink (not drowsy)
BLINK_WINDOW_S       = 60.0     # rolling window for blink-rate calculation
LOW_BLINK_RATE       = 8        # blinks/min below this = fatigue indicator

# MAR / yawns
MAR_THRESHOLD        = 0.60
MAR_SMOOTH_LEN       = 5
YAWN_BURST_COUNT     = 3        # yawns within window → burst alert
YAWN_BURST_WINDOW    = 60.0     # seconds

# PERCLOS
PERCLOS_WINDOW_S     = 30       # rolling seconds
PERCLOS_ALERT        = 0.25     # ≥25 % closed → FATIGUED

# Calibration
CALIBRATION_DURATION = 5        # seconds

# Alerts
DROWSY_ALERT_COOLDOWN = 4.0
YAWN_ALERT_COOLDOWN   = 10.0

# Audio
SAMPLE_RATE          = 44100
MASTER_VOLUME        = 0.9

# Risk score weights  (must sum to 100)
RISK_W_PERCLOS       = 35
RISK_W_BLINK_RATE    = 20
RISK_W_YAWN_RATE     = 20
RISK_W_EPISODES      = 25

# Misc
BREAK_REMINDER_EPISODES = 3     # suggest break after N drowsy episodes

#LANDMARK INDICES

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308]   # top, bottom, left-corner, right-corner

#AUDIO ALERT

_pygame_ok = False
_mixer_channels = 1
try:
    import pygame
    import pygame.sndarray
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)   
    _freq, _size, _mixer_channels = pygame.mixer.get_init()
    _pygame_ok = True
except Exception as exc:
    print()


def _sine_wave(freq: float, dur_s: float,
               vol: float = 1.0, fade_ms: int = 8) -> np.ndarray:
    n      = max(1, int(SAMPLE_RATE * dur_s))
    t      = np.linspace(0, dur_s, n, endpoint=False)
    wave   = np.sin(2.0 * np.pi * freq * t)
    fade_n = min(int(SAMPLE_RATE * fade_ms / 1000), n // 4)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)
        wave[:fade_n]  *= ramp
        wave[-fade_n:] *= ramp[::-1]
    mono = (wave * vol * MASTER_VOLUME * 32767).astype(np.int16)
    if _mixer_channels >= 2:
        return np.column_stack([mono, mono])
    return mono


def _play_sequence(segments: list) -> None:
    if not _pygame_ok:
        if platform.system() == "Windows":
            import winsound
            for freq, dur, gap in segments:
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                if gap:
                    time.sleep(gap)
        else:
            for _ in segments:
                print("\a", end="", flush=True)
                time.sleep(0.15)
        return
    try:
        for freq, dur, gap in segments:
            sound = pygame.sndarray.make_sound(_sine_wave(freq, dur))
            sound.play()
            time.sleep(dur + gap)
    except Exception as exc:
        print()


#Drowsy
_DROWSY_SEQ = [
    (1500, 0.14, 0.09),
    (1550, 0.12, 0.08),
    (1600, 0.10, 0.07),
    (1700, 0.08, 0.05),
    (1800, 0.07, 0.04),
    (1900, 0.06, 0.03),
    (2000, 0.70, 0.00),
]

#Yawn
_YAWN_SEQ = [
    (880, 0.30, 0.12),
    (660, 0.30, 0.12),
    (440, 0.50, 0.00),
]

_drown_lock = threading.Lock()
_yawn_lock  = threading.Lock()
_last_drown = 0.0
_last_yawn  = 0.0


def play_drowsy_alert() -> None:
    global _last_drown
    now = time.time()
    with _drown_lock:
        if now - _last_drown < DROWSY_ALERT_COOLDOWN:
            return
        _last_drown = now
    threading.Thread(target=_play_sequence, args=(_DROWSY_SEQ,), daemon=True).start()


def play_yawn_alert() -> None:
    global _last_yawn
    now = time.time()
    with _yawn_lock:
        if now - _last_yawn < YAWN_ALERT_COOLDOWN:
            return
        _last_yawn = now
    threading.Thread(target=_play_sequence, args=(_YAWN_SEQ,), daemon=True).start()


#GEOMETRY UTILITIES

def euclidean(p1, p2) -> float:
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_EAR(eye) -> float:
    v1 = euclidean(eye[1], eye[5])
    v2 = euclidean(eye[2], eye[4])
    h  = euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_MAR(mouth) -> float:
    v = euclidean(mouth[0], mouth[1])
    h = euclidean(mouth[2], mouth[3])
    return v / h if h > 0 else 0.0


def risk_score(perclos: float, blink_rate: float,
               yawns_recent: int, episodes: int) -> int:
    p = min(perclos / 0.50, 1.0)                          # 50 % PERCLOS = max
    b = max(0.0, min(1.0 - blink_rate / LOW_BLINK_RATE, 1.0))  # below LOW = bad
    y = min(yawns_recent / (YAWN_BURST_COUNT * 2), 1.0)   # 2× burst = max
    e = min(episodes / (BREAK_REMINDER_EPISODES * 2), 1.0)

    score = (p * RISK_W_PERCLOS +
             b * RISK_W_BLINK_RATE +
             y * RISK_W_YAWN_RATE +
             e * RISK_W_EPISODES)
    return int(round(score))


#HUD DRAWING

def draw_bar(frame, label: str, value: float, max_val: float,
             x: int, y: int, w: int, color: tuple) -> None:
    filled = int(min(value / max_val, 1.0) * w) if max_val > 0 else 0
    cv2.rectangle(frame, (x, y), (x + w, y + 13), (45, 45, 45), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + 13), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1)


def draw_risk_gauge(frame, score: int, x: int, y: int, w: int) -> None:
    ratio = score / 100.0
    if ratio < 0.4:
        col = (0, 200, 0)
    elif ratio < 0.7:
        t   = (ratio - 0.4) / 0.3
        col = (0, int(200 * (1 - t)), int(220 * t))
    else:
        col = (0, 30, 220)

    filled = int(ratio * w)
    cv2.rectangle(frame, (x, y), (x + w, y + 16), (45, 45, 45), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + 16), col, -1)
    label = f"RISK: {score}/100"
    cv2.putText(frame, label, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)


def overlay_dashboard(frame, stats: dict) -> None:
    fh, fw = frame.shape[:2]
    panel_w = 235

    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (panel_w, fh), (15, 15, 15), -1)
    cv2.addWeighted(bg, 0.62, frame, 0.38, 0, frame)

    # Title
    cv2.putText(frame, "DRIVER MONITOR", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 200, 255), 2)
    cv2.line(frame, (8, 30), (panel_w - 8, 30), (0, 200, 255), 1)

    ear_thresh = stats.get("ear_thresh", 0.2)
    ear_col  = (0, 210, 0)   if stats["ear"] > ear_thresh      else (0, 50, 230)
    mar_col  = (0, 210, 255) if stats["mar"] < MAR_THRESHOLD   else (0, 150, 255)
    perc_col = (0, 210, 0)   if stats["perclos"] < PERCLOS_ALERT else (0, 50, 230)
    blnk_col = (0, 210, 0)   if stats["blink_rate"] >= LOW_BLINK_RATE else (0, 150, 255)

    draw_bar(frame, "EAR",        stats["ear"],        0.45, 8,  50, 200, ear_col)
    draw_bar(frame, "MAR",        stats["mar"],        1.20, 8,  82, 200, mar_col)
    draw_bar(frame, "PERCLOS",    stats["perclos"],    0.50, 8, 114, 200, perc_col)
    draw_bar(frame, "BLINK/min",  stats["blink_rate"], 30,   8, 146, 200, blnk_col)
    draw_risk_gauge(frame,        stats["risk"],             8, 178, 200)

    cv2.line(frame, (8, 202), (panel_w - 8, 202), (60, 60, 60), 1)

    elapsed_str = time.strftime("%M:%S", time.gmtime(stats["elapsed"]))
    rows = [
        ("Session",   elapsed_str),
        ("Yawns",     str(stats["yawns"])),
        ("Yawn/60s",  str(stats["yawns_recent"])),
        ("Episodes",  str(stats["episodes"])),
        ("Blinks",    str(stats["total_blinks"])),
        ("FPS",       f"{stats['fps']:.0f}"),
    ]
    y0 = 218
    for label, val in rows:
        cv2.putText(frame, f"{label:<10}{val}", (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (195, 195, 195), 1)
        y0 += 21

    # STATUS BADGE
    status = stats["status"]
    badge_col = {
        "ALERT":       (0,   0, 220),
        "YAWN ALERT":  (0,  55, 200),
        "FATIGUED":    (0,  90, 200),
        "YAWNING":     (0, 170, 255),
        "LOW BLINKS":  (0, 130, 210),
        "OK":          (0, 150,   0),
        "CALIBRATING": (0, 220, 220),
        "NO FACE":     (80,  80,  80),
    }.get(status, (100, 100, 100))
    cv2.rectangle(frame, (8, fh - 46), (panel_w - 8, fh - 10), badge_col, -1)
    fs = 0.65 if len(status) <= 8 else 0.50
    cv2.putText(frame, status, (14, fh - 22),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)

    # BREAK REMINDER BANNER
    if stats.get("show_break_reminder"):
        bx, by = panel_w + 10, fh - 50
        cv2.rectangle(frame, (bx, by), (fw - 10, fh - 10), (0, 0, 180), -1)
        cv2.putText(frame, "REST RECOMMENDED — pull over safely",
                    (bx + 8, fh - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_calib_progress(frame, fraction: float) -> None:
    fh, fw = frame.shape[:2]
    bar_x, bar_y, bar_w, bar_h = fw // 4, 12, fw // 2, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled = int(fraction * bar_w)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 220, 220), -1)
    cv2.putText(frame, f"CALIBRATING... {int(fraction * 100)}%",
                (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)


def draw_face_overlays(frame, left_eye, right_eye, mouth_pts) -> None:
    cv2.polylines(frame, [left_eye],  True, (0, 230, 0),   1)
    cv2.polylines(frame, [right_eye], True, (0, 230, 0),   1)
    cv2.polylines(frame, [mouth_pts], True, (0, 200, 255), 1)



def main() -> None:
    #CAMERA SETUP
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    #MEDIAPIPE
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    #ROLLING BUFFERS
    ear_buffer     = deque(maxlen=EAR_SMOOTH_LEN)
    mar_buffer     = deque(maxlen=MAR_SMOOTH_LEN)
    perclos_deque  = deque()    # (timestamp, is_closed)
    blink_deque    = deque()    # timestamps of blink completions (rolling 60 s)

    #CALIBRATION STATE
    calib_values   = []
    calib_start    = time.time()
    calibrated     = False
    EAR_THRESHOLD  = None       # set after calibration

    #DROWSINESS STATE
    drowsy_start_t       = None
    in_drowsy_episode    = False
    drowsy_episode_count = 0
    microsleep_durations = []       # list of seconds per episode
    episode_start_t      = None

    #BLINK STATE
    eye_closed_start = None         # when this closure began
    total_blinks     = 0

    #YAWN STATE
    yawn_count       = 0
    yawn_timestamps  = deque()      # rolling 60 s
    prev_mar_state   = False

    #DISPLAY STATE
    ear_val      = 0.0
    mar_val      = 0.0
    perclos      = 0.0
    blink_rate   = 15.0             # sensible starting value
    yawns_recent = 0

    #SUMMARY STATS
    session_start    = time.time()
    perclos_samples  = []           # for accurate session-average PERCLOS

    print(f"[INFO] Keep eyes open and face the camera — "
          f"calibrating for {CALIBRATION_DURATION} s.")

    try:
        while cap.isOpened():
            now = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            if CAM_FLIP:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            status           = "OK"
            face_present     = bool(res.multi_face_landmarks)

            #FACE ABSENT
            if not face_present:
                # Drowsy timer must not continue without face confirmation
                drowsy_start_t    = None
                in_drowsy_episode = False
                eye_closed_start  = None
                status            = "NO FACE"
                # Still prune time-based deques
                while yawn_timestamps and (now - yawn_timestamps[0]) > YAWN_BURST_WINDOW:
                    yawn_timestamps.popleft()
                while blink_deque and (now - blink_deque[0]) > BLINK_WINDOW_S:
                    blink_deque.popleft()
                yawns_recent = len(yawn_timestamps)
                cv2.putText(frame, "No face detected",
                            (fw // 2 - 140, fh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

            else:
                lm = np.array(
                    [[int(p.x * fw), int(p.y * fh)]
                     for p in res.multi_face_landmarks[0].landmark]
                )

                left_eye  = lm[LEFT_EYE]
                right_eye = lm[RIGHT_EYE]
                mouth_pts = lm[MOUTH]

                raw_ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
                raw_mar = calculate_MAR(mouth_pts)

                #CALIBRATION
                if not calibrated:
                    elapsed_c = now - calib_start
                    progress  = min(elapsed_c / CALIBRATION_DURATION, 1.0)

                    if elapsed_c < CALIBRATION_DURATION:
                        calib_values.append(raw_ear)
                        status = "CALIBRATING"
                        draw_calib_progress(frame, progress)
                    else:
                        # Guard against empty list (face was never detected)
                        if len(calib_values) < 5:
                            # Restart calibration
                            calib_values = []
                            calib_start  = now
                            print("[WARN] Not enough calibration data — restarting.")
                        else:
                            baseline_ear    = float(np.mean(calib_values))
                            EAR_THRESHOLD   = max(baseline_ear * EAR_CALIBRATION_FACTOR,
                                                  0.12)
                            calibrated      = True
                            print(f"[INFO] Calibration complete. "
                                  f"EAR baseline={baseline_ear:.3f}  "
                                  f"threshold={EAR_THRESHOLD:.3f}")

                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    overlay_dashboard(frame, {
                        "ear": raw_ear, "mar": raw_mar, "perclos": 0.0,
                        "blink_rate": 15.0, "risk": 0,
                        "yawns": 0, "yawns_recent": 0, "episodes": 0,
                        "total_blinks": 0,
                        "elapsed": now - session_start,
                        "status": status, "fps": fps, "ear_thresh": 0.2,
                        "show_break_reminder": False,
                    })
                    cv2.imshow("Advanced Driver Monitor", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q') or key == 27:
                        break
                    continue

                #SMOOTHING
                ear_buffer.append(raw_ear)
                mar_buffer.append(raw_mar)
                ear_val = float(np.mean(ear_buffer))
                mar_val = float(np.mean(mar_buffer))

                is_closed = ear_val < EAR_THRESHOLD

                #BLINK DETECTION
                # A blink = eye closes then reopens within BLINK_REOPEN_MAX_S
                if is_closed:
                    if eye_closed_start is None:
                        eye_closed_start = now   # leading edge
                else:
                    if eye_closed_start is not None:
                        duration = now - eye_closed_start
                        if duration <= BLINK_REOPEN_MAX_S:
                            # Genuine blink (not a drowsy closure)
                            total_blinks += 1
                            blink_deque.append(now)
                        eye_closed_start = None  # trailing edge — reset regardless

                # Prune blink window
                while blink_deque and (now - blink_deque[0]) > BLINK_WINDOW_S:
                    blink_deque.popleft()
                blink_rate = len(blink_deque) * (60.0 / BLINK_WINDOW_S)

                #PERCLOS
                perclos_deque.append((now, is_closed))
                while perclos_deque and (now - perclos_deque[0][0]) > PERCLOS_WINDOW_S:
                    perclos_deque.popleft()
                if perclos_deque:
                    perclos = sum(1 for _, c in perclos_deque if c) / len(perclos_deque)
                perclos_samples.append(perclos)

                #DROWSINESS DETECTION
                if is_closed:
                    if drowsy_start_t is None:
                        drowsy_start_t  = now
                        episode_start_t = now
                    elif now - drowsy_start_t > DROWSY_TIME_THRESHOLD:
                        status = "ALERT"
                        if not in_drowsy_episode:
                            drowsy_episode_count += 1
                            in_drowsy_episode = True
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 140), -1)
                        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
                        cv2.putText(frame, "!  DROWSY  !",
                                    (fw // 2 - 155, fh // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                        play_drowsy_alert()
                else:
                    if in_drowsy_episode and episode_start_t is not None:
                        microsleep_durations.append(now - episode_start_t)
                    drowsy_start_t    = None
                    in_drowsy_episode = False
                    episode_start_t   = None

                # PERCLOS fatigue flag
                if perclos >= PERCLOS_ALERT and status == "OK":
                    status = "FATIGUED"

                # Low blink rate flag
                if blink_rate < LOW_BLINK_RATE and status == "OK":
                    status = "LOW BLINKS"

                #YAWN DETECTION
                mouth_open = mar_val > MAR_THRESHOLD
                if mouth_open and not prev_mar_state:
                    yawn_count += 1
                    yawn_timestamps.append(now)

                while yawn_timestamps and (now - yawn_timestamps[0]) > YAWN_BURST_WINDOW:
                    yawn_timestamps.popleft()
                yawns_recent = len(yawn_timestamps)

                if yawns_recent >= YAWN_BURST_COUNT:
                    if status in ("OK", "YAWNING", "LOW BLINKS", "FATIGUED"):
                        status = "YAWN ALERT"
                    cv2.putText(frame,
                                f"TAKE A BREAK!  {yawns_recent} yawns / 60 s",
                                (fw // 2 - 225, fh // 2 + 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 60, 255), 2)
                    play_yawn_alert()

                prev_mar_state = mouth_open
                if mouth_open and status == "OK":
                    status = "YAWNING"

                draw_face_overlays(frame, left_eye, right_eye, mouth_pts)

            #RISK SCORE
            r_score = risk_score(perclos, blink_rate, yawns_recent,
                                 drowsy_episode_count)

            #BREAK REMINDER
            show_break = (drowsy_episode_count >= BREAK_REMINDER_EPISODES or
                          r_score >= 70)

            #DASHBOARD
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            overlay_dashboard(frame, {
                "ear":               ear_val,
                "mar":               mar_val,
                "perclos":           perclos,
                "blink_rate":        blink_rate,
                "risk":              r_score,
                "yawns":             yawn_count,
                "yawns_recent":      yawns_recent,
                "episodes":          drowsy_episode_count,
                "total_blinks":      total_blinks,
                "elapsed":           now - session_start,
                "status":            status,
                "fps":               fps,
                "ear_thresh":        EAR_THRESHOLD or 0.2,
                "show_break_reminder": show_break,
            })

            cv2.imshow("Advanced Driver Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process pending GUI events to actually close the window
        face_mesh.close()
        if _pygame_ok:
            pygame.mixer.quit()

        elapsed = time.time() - session_start
        avg_perclos = float(np.mean(perclos_samples)) if perclos_samples else 0.0
        avg_ms = (float(np.mean(microsleep_durations))
                  if microsleep_durations else 0.0)
        sys.exit(0)


if __name__ == "__main__":
    main()