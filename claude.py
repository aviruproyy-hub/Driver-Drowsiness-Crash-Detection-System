"""
Advanced Driver Drowsiness Monitor
====================================
Audio alert system:
  - DROWSY  : Aggressive escalating siren burst (rapid high-pitched beeps, gets faster)
  - YAWN×3  : Distinct advisory chime (descending 3-tone melody) when ≥3 yawns in 60 s
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import platform
import threading
from collections import deque
from math import sqrt

# ============================================================
#  CONFIG — Tune these without touching logic below
# ============================================================

EAR_CALIBRATION_FACTOR   = 0.75   # EAR threshold = baseline × this
MAR_THRESHOLD            = 0.60   # Mouth Aspect Ratio for yawn
DROWSY_TIME_THRESHOLD    = 1.5    # Seconds of low EAR before DROWSY alert
HEAD_DROP_THRESHOLD      = 30     # Pixel drop in nose-Y to flag head nodding
HEAD_NOD_THRESHOLD       = 20     # Pixel rise (forward nod toward camera)
PERCLOS_WINDOW           = 30     # Rolling seconds for PERCLOS calculation
PERCLOS_ALERT_THRESHOLD  = 0.25   # 25% eye-closure = FATIGUED
CALIBRATION_DURATION     = 5      # Seconds of open-eye calibration
EAR_SMOOTH_LEN           = 10     # Frames for EAR rolling average
MAR_SMOOTH_LEN           = 5      # Frames for MAR rolling average

YAWN_BURST_COUNT         = 3      # Yawns within window that trigger burst alert
YAWN_BURST_WINDOW        = 60.0   # Rolling window in seconds

DROWSY_ALERT_COOLDOWN    = 4.0    # Seconds between repeated drowsy alerts
YAWN_ALERT_COOLDOWN      = 10.0   # Seconds between repeated yawn-burst alerts

SAMPLE_RATE              = 44100  # Audio sample rate (Hz)
MASTER_VOLUME            = 0.9    # 0.0 to 1.0

# ============================================================
#  Landmark Indices  (MediaPipe 468-point Face Mesh)
# ============================================================

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308]
NOSE_TIP  = 1
CHIN      = 152

# ============================================================
#  Audio Engine
# ============================================================

_pygame_ok = False
try:
    import pygame
    import pygame.sndarray
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
    pygame.mixer.init()
    _pygame_ok = True
    print("[AUDIO] pygame mixer initialised OK")
except Exception as e:
    print(f"[AUDIO] pygame unavailable ({e}). Falling back to system beep.")


def _sine_wave(freq: float, duration_s: float,
               volume: float = 1.0, fade_ms: int = 8) -> np.ndarray:
    """Generate a mono 16-bit numpy sine-wave buffer."""
    n    = int(SAMPLE_RATE * duration_s)
    t    = np.linspace(0, duration_s, n, endpoint=False)
    wave = np.sin(2.0 * np.pi * freq * t)

    # Tiny fade in/out to eliminate clicks
    fade_n = min(int(SAMPLE_RATE * fade_ms / 1000), n // 4)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)
        wave[:fade_n]  *= ramp
        wave[-fade_n:] *= ramp[::-1]

    return (wave * volume * MASTER_VOLUME * 32767).astype(np.int16)


def _play_sequence(segments):
    """
    Play a sequence of (freq_Hz, duration_s, gap_after_s) tuples.
    Blocking — always call from a daemon thread.
    """
    if not _pygame_ok:
        if platform.system() == "Windows":
            import winsound
            for freq, dur, gap in segments:
                winsound.Beep(int(freq), int(dur * 1000))
                if gap:
                    time.sleep(gap)
        else:
            for _ in segments:
                print("\a", end="", flush=True)
                time.sleep(0.15)
        return

    for freq, dur, gap in segments:
        buf   = _sine_wave(freq, dur)
        sound = pygame.sndarray.make_sound(buf)
        sound.play()
        time.sleep(dur + gap)


# ── Drowsy alert: escalating siren ──────────────────────────────────
# Six rapid high beeps, each shorter than the last (urgency increases),
# ending with a long 700 ms blast. Nothing like a gentle single beep.
_DROWSY_SEQ = [
    (1500, 0.14, 0.09),   # beep 1
    (1550, 0.12, 0.08),   # beep 2
    (1600, 0.10, 0.07),   # beep 3
    (1700, 0.08, 0.05),   # beep 4 — faster now
    (1800, 0.07, 0.04),   # beep 5
    (1900, 0.06, 0.03),   # beep 6
    (2000, 0.70, 0.00),   # long blaring klaxon
]

# ── Yawn-burst alert: descending advisory chime ──────────────────────
# Three descending tones — calm but unmistakably different from the
# siren, communicating "you are getting tired, pull over" rather than
# "WAKE UP RIGHT NOW".
_YAWN_BURST_SEQ = [
    (880, 0.30, 0.12),   # high note
    (660, 0.30, 0.12),   # mid note
    (440, 0.50, 0.00),   # low resolution note
]

_drowsy_lock       = threading.Lock()
_yawn_lock         = threading.Lock()
_last_drowsy_t     = 0.0
_last_yawn_burst_t = 0.0


def play_drowsy_alert():
    """Non-blocking escalating siren burst."""
    global _last_drowsy_t
    now = time.time()
    with _drowsy_lock:
        if now - _last_drowsy_t < DROWSY_ALERT_COOLDOWN:
            return
        _last_drowsy_t = now
    threading.Thread(target=_play_sequence,
                     args=(_DROWSY_SEQ,), daemon=True).start()


def play_yawn_burst_alert():
    """Non-blocking descending advisory chime."""
    global _last_yawn_burst_t
    now = time.time()
    with _yawn_lock:
        if now - _last_yawn_burst_t < YAWN_ALERT_COOLDOWN:
            return
        _last_yawn_burst_t = now
    threading.Thread(target=_play_sequence,
                     args=(_YAWN_BURST_SEQ,), daemon=True).start()


# ============================================================
#  Geometry Utilities
# ============================================================

def euclidean(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_EAR(eye):
    """Eye Aspect Ratio — Soukupova & Cech (2016)."""
    v1 = euclidean(eye[1], eye[5])
    v2 = euclidean(eye[2], eye[4])
    h  = euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_MAR(mouth):
    """Mouth Aspect Ratio."""
    v = euclidean(mouth[0], mouth[1])
    h = euclidean(mouth[2], mouth[3])
    return v / h if h > 0 else 0.0


# ============================================================
#  HUD Drawing
# ============================================================

def draw_bar(frame, label, value, max_val, x, y, bar_w, color):
    filled = int(min(value / max_val, 1.0) * bar_w)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 14), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled,  y + 14), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def overlay_dashboard(frame, stats: dict):
    fh, fw = frame.shape[:2]
    panel_w = 230

    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (panel_w, fh), (18, 18, 18), -1)
    cv2.addWeighted(bg, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, "DRIVER MONITOR", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.line(frame, (8, 32), (panel_w - 8, 32), (0, 200, 255), 1)

    ear_col  = (0, 210, 0)   if stats["ear"] > stats.get("ear_thresh", 0.2) else (0, 60, 255)
    mar_col  = (0, 210, 255) if stats["mar"] < MAR_THRESHOLD else (0, 140, 255)
    perc_col = (0, 210, 0)   if stats["perclos"] < PERCLOS_ALERT_THRESHOLD else (0, 60, 255)

    draw_bar(frame, "EAR",     stats["ear"],     0.45, 8, 55,  200, ear_col)
    draw_bar(frame, "MAR",     stats["mar"],     1.20, 8, 90,  200, mar_col)
    draw_bar(frame, "PERCLOS", stats["perclos"], 0.50, 8, 125, 200, perc_col)

    elapsed_str = time.strftime("%M:%S", time.gmtime(stats["elapsed"]))
    rows = [
        ("Session",    elapsed_str),
        ("Yawns",      str(stats["yawns"])),
        ("Yawn/60s",   str(stats["yawns_recent"])),
        ("Episodes",   str(stats["episodes"])),
        ("FPS",        f"{stats['fps']:.0f}"),
    ]
    y0 = 165
    for label, val in rows:
        cv2.putText(frame, f"{label:<10}{val}", (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        y0 += 22

    status = stats["status"]
    badge_color = {
        "ALERT":       (0,  0,  220),
        "YAWN ALERT":  (0,  60, 200),
        "FATIGUED":    (0,  90, 200),
        "YAWNING":     (0,  180, 255),
        "OK":          (0,  150,  0),
        "CALIBRATING": (0,  220, 220),
        "NO FACE":     (90,  90,  90),
    }.get(status, (110, 110, 110))

    cv2.rectangle(frame, (8, fh - 48), (panel_w - 8, fh - 10), badge_color, -1)
    fs = 0.65 if len(status) <= 8 else 0.52
    cv2.putText(frame, status, (14, fh - 24),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)


def draw_eye_contours(frame, left_eye, right_eye, mouth):
    cv2.polylines(frame, [left_eye],  True, (0, 230, 0),   1)
    cv2.polylines(frame, [right_eye], True, (0, 230, 0),   1)
    cv2.polylines(frame, [mouth],     True, (0, 200, 255), 1)


# ============================================================
#  Main
# ============================================================

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    EAR_THRESHOLD        = None
    ear_buffer           = deque(maxlen=EAR_SMOOTH_LEN)
    mar_buffer           = deque(maxlen=MAR_SMOOTH_LEN)
    perclos_window       = deque()

    drowsy_start_time    = None
    drowsy_episode_count = 0
    in_drowsy_episode    = False

    baseline_nose_y      = None
    baseline_chin_y      = None

    yawn_count           = 0
    yawn_timestamps      = deque()   # timestamps for rolling 60 s window
    prev_mar_state       = False

    calib_values         = []
    calib_start          = time.time()
    calibrated           = False
    session_start        = time.time()
    perclos              = 0.0

    print(f"[INFO] Calibrating for {CALIBRATION_DURATION}s — keep eyes open, face forward.")

    try:
        while cap.isOpened():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            status       = "OK"
            ear_val      = 0.0
            mar_val      = 0.0
            perclos      = 0.0
            yawns_recent = len(yawn_timestamps)

            if res.multi_face_landmarks:
                lm = np.array(
                    [[int(p.x * fw), int(p.y * fh)]
                     for p in res.multi_face_landmarks[0].landmark]
                )

                left_eye  = lm[LEFT_EYE]
                right_eye = lm[RIGHT_EYE]
                mouth_pts = lm[MOUTH]
                nose      = lm[NOSE_TIP]
                chin      = lm[CHIN]

                raw_ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
                raw_mar = calculate_MAR(mouth_pts)

                # ── CALIBRATION ───────────────────────────────────────────
                if not calibrated:
                    ec = time.time() - calib_start
                    if ec < CALIBRATION_DURATION:
                        calib_values.append(raw_ear)
                        status = "CALIBRATING"
                    else:
                        baseline_ear    = float(np.mean(calib_values))
                        EAR_THRESHOLD   = baseline_ear * EAR_CALIBRATION_FACTOR
                        baseline_nose_y = int(nose[1])
                        baseline_chin_y = int(chin[1])
                        calibrated      = True
                        print(f"[INFO] Calibration done.  "
                              f"Baseline EAR={baseline_ear:.3f}  "
                              f"Threshold={EAR_THRESHOLD:.3f}")

                    fps = 1.0 / max(time.time() - t0, 1e-6)
                    overlay_dashboard(frame, {
                        "ear": raw_ear, "mar": raw_mar, "perclos": 0,
                        "yawns": 0, "yawns_recent": 0, "episodes": 0,
                        "elapsed": time.time() - session_start,
                        "status": status, "fps": fps, "ear_thresh": 0.2,
                    })
                    cv2.imshow("Advanced Driver Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # ── SMOOTHING ─────────────────────────────────────────────
                ear_buffer.append(raw_ear)
                mar_buffer.append(raw_mar)
                ear_val = float(np.mean(ear_buffer))
                mar_val = float(np.mean(mar_buffer))

                # ── PERCLOS ───────────────────────────────────────────────
                now       = time.time()
                is_closed = ear_val < EAR_THRESHOLD
                perclos_window.append((now, is_closed))
                while perclos_window and (now - perclos_window[0][0]) > PERCLOS_WINDOW:
                    perclos_window.popleft()
                if perclos_window:
                    perclos = sum(1 for _, c in perclos_window if c) / len(perclos_window)

                # ── DROWSINESS DETECTION ──────────────────────────────────
                if is_closed:
                    if drowsy_start_time is None:
                        drowsy_start_time = now
                    elif now - drowsy_start_time > DROWSY_TIME_THRESHOLD:
                        status = "ALERT"
                        if not in_drowsy_episode:
                            drowsy_episode_count += 1
                            in_drowsy_episode = True
                        # Red tint on frame
                        red = frame.copy()
                        cv2.rectangle(red, (0, 0), (fw, fh), (0, 0, 160), -1)
                        cv2.addWeighted(red, 0.18, frame, 0.82, 0, frame)
                        cv2.putText(frame, "!  DROWSY  !",
                                    (fw // 2 - 150, fh // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                        play_drowsy_alert()       # <-- siren burst
                else:
                    drowsy_start_time = None
                    in_drowsy_episode = False

                if perclos >= PERCLOS_ALERT_THRESHOLD and status == "OK":
                    status = "FATIGUED"

                # ── YAWN DETECTION + BURST TRACKING ──────────────────────
                mouth_open = mar_val > MAR_THRESHOLD

                if mouth_open and not prev_mar_state:
                    # Leading edge = new yawn
                    yawn_count += 1
                    yawn_timestamps.append(now)

                # Keep only the last 60 s of yawn timestamps
                while yawn_timestamps and (now - yawn_timestamps[0]) > YAWN_BURST_WINDOW:
                    yawn_timestamps.popleft()

                yawns_recent = len(yawn_timestamps)

                if yawns_recent >= YAWN_BURST_COUNT:
                    if status in ("OK", "YAWNING"):
                        status = "YAWN ALERT"
                    cv2.putText(frame,
                                f"TAKE A BREAK!  {yawns_recent} yawns in 60s",
                                (fw // 2 - 215, fh // 2 + 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 70, 255), 2)
                    play_yawn_burst_alert()        # <-- descending chime

                prev_mar_state = mouth_open
                if mouth_open and status == "OK":
                    status = "YAWNING"

                # ── HEAD DROP / FORWARD NOD ───────────────────────────────
                nose_dy = int(nose[1]) - baseline_nose_y
                chin_dy = int(chin[1]) - baseline_chin_y

                if nose_dy > HEAD_DROP_THRESHOLD:
                    cv2.putText(frame, "v  HEAD DROP", (fw // 2 - 110, fh - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 40, 255), 2)
                    if status == "OK":
                        status = "ALERT"
                    play_drowsy_alert()

                if chin_dy < -HEAD_NOD_THRESHOLD:
                    cv2.putText(frame, "^  HEAD NOD", (fw // 2 - 110, fh - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 100, 255), 2)
                    if status == "OK":
                        status = "ALERT"

                draw_eye_contours(frame, left_eye, right_eye, mouth_pts)

            else:
                status       = "NO FACE"
                yawns_recent = len(yawn_timestamps)
                cv2.putText(frame, "No face detected",
                            (fw // 2 - 130, fh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

            # ── DASHBOARD ─────────────────────────────────────────────────
            fps = 1.0 / max(time.time() - t0, 1e-6)
            overlay_dashboard(frame, {
                "ear":          ear_val,
                "mar":          mar_val,
                "perclos":      perclos,
                "yawns":        yawn_count,
                "yawns_recent": yawns_recent,
                "episodes":     drowsy_episode_count,
                "elapsed":      time.time() - session_start,
                "status":       status,
                "fps":          fps,
                "ear_thresh":   EAR_THRESHOLD or 0.2,
            })

            cv2.imshow("Advanced Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        if _pygame_ok:
            pygame.mixer.quit()

        elapsed = time.time() - session_start
        print("\n========== SESSION SUMMARY ==========")
        print(f"  Duration       : {time.strftime('%M:%S', time.gmtime(elapsed))}")
        print(f"  Drowsy episodes: {drowsy_episode_count}")
        print(f"  Total yawns    : {yawn_count}")
        print(f"  Final PERCLOS  : {perclos:.1%}")
        print("=====================================\n")


if __name__ == "__main__":
    main()