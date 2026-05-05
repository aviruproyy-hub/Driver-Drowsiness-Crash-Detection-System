"""Drowsiness & Yawn Detection with MediaPipe. Press Q to quit."""
import sys, os, urllib.request, threading, socket, json, time, platform
import cv2, numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image_processing_options import ImageProcessingOptions

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
def download_model():
    if os.path.exists(MODEL_PATH):
        return True
    print("[Drowsiness] Downloading model (~30 MB)...")
    try:
        def _progress(block_num, block_size, total_size):
            pct = min(100 if total_size <= 0 else block_num * block_size / total_size * 100, 100)
            print(f"\r  [{int(pct / 5) * '#'}{' ' * (20 - int(pct / 5))}] {pct:.0f}%", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _progress)
        print("\n[Drowsiness] Downloaded successfully.")
        return True
    except Exception as e:
        print(f"\n[Drowsiness] Failed: {e}\nManually download and save as face_landmarker.task")
        return False
if not download_model():
    sys.exit(1)
def play_sound(frequency, duration_ms):
    def _beep():
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(frequency, duration_ms)
            else:
                import subprocess
                subprocess.run(["beep", f"-f {frequency}", f"-l {duration_ms}"], stderr=subprocess.DEVNULL)
        except:
            print("\a", end="", flush=True)
    threading.Thread(target=_beep, daemon=True).start()
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]
def calculate_EAR(pts):
    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    h = dist.euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
def calculate_MAR(pts):
    v = dist.euclidean(pts[0], pts[1])
    h = dist.euclidean(pts[2], pts[3])
    return v / h if h > 0 else 0.0
_latest_result = None
_result_lock = threading.Lock()
_latest_ts = 0
def _on_result(result: FaceLandmarkerResult, output_image: MpImage, ts: int):
    global _latest_result, _latest_ts
    with _result_lock:
        _latest_result = result
        _latest_ts = ts
crash_alert_active = False
crash_alert_msg = ""
crash_alert_lock = threading.Lock()
def crash_listener(port=65432):
    global crash_alert_active, crash_alert_msg
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", port))
        server.listen(1)
        server.settimeout(2)
        print(f"[Drowsiness] IPC listening on port {port}")
        while True:
            try:
                conn, _ = server.accept()
                data = conn.recv(1024).decode()
                conn.close()
                if data:
                    payload = json.loads(data)
                    with crash_alert_lock:
                        crash_alert_active = payload.get("crash", False)
                        crash_alert_msg = payload.get("message", "CRASH DETECTED")
            except:
                continue
    except Exception as e:
        print(f"[Drowsiness] IPC error: {e}")
    finally:
        server.close()
threading.Thread(target=crash_listener, daemon=True).start()
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=_on_result,
)
landmarker = FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.50
FRAME_CHECK = 20
EYE_COUNTER = 0
YAWN_COUNT = 0
YAWN_COOLDOWN = False
YAWN_ALERT_DONE = False
SESSION_START = time.time()
frame_index = 0
print("[Drowsiness] Running. Press Q to quit.")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp)
    with _result_lock:
        result = _latest_result
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.line(frame, (0, 80), (w, 80), (0, 200, 80), 2)
    elapsed = int(time.time() - SESSION_START)
    mins, sec = divmod(elapsed, 60)
    cv2.putText(frame, f"SESSION {mins:02d}:{sec:02d}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "ADAS v2.0 | DROWSINESS", (w - 280, 25), cv2.FONT_HERSHEY_DUPLEX, 0.45, (80, 80, 80), 1)
    status_text = "AWAKE"
    status_color = (0, 220, 80)
    fill_level = 0.0
    if result and result.face_landmarks:
        for face_lm in result.face_landmarks:
            lm = np.array([[int(p.x * w), int(p.y * h)] for p in face_lm])
            left_eye_pts = lm[LEFT_EYE]
            right_eye_pts = lm[RIGHT_EYE]
            mouth_pts = lm[MOUTH]
            ear = (calculate_EAR(left_eye_pts) + calculate_EAR(right_eye_pts)) / 2.0
            mar = calculate_MAR(mouth_pts)
            eye_color = (0, 220, 80) if ear >= EAR_THRESHOLD else (0, 60, 255)
            mouth_color = (0, 220, 220) if mar < MAR_THRESHOLD else (0, 140, 255)
            cv2.polylines(frame, [left_eye_pts], True, eye_color, 1)
            cv2.polylines(frame, [right_eye_pts], True, eye_color, 1)
            cv2.polylines(frame, [mouth_pts], True, mouth_color, 1)
            cv2.putText(frame, f"EAR:{ear:.2f}", (10, h - 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"MAR:{mar:.2f}", (10, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
            if ear < EAR_THRESHOLD:
                EYE_COUNTER += 1
            elif EYE_COUNTER > 0:
                EYE_COUNTER -= 1
            if mar > MAR_THRESHOLD:
                if not YAWN_COOLDOWN:
                    YAWN_COUNT += 1
                    YAWN_COOLDOWN = True
            else:
                YAWN_COOLDOWN = False
            fill_level = min(EYE_COUNTER / FRAME_CHECK, 1.0)
            bar_total = 210
            bar_filled = int(fill_level * bar_total)
            bar_color = (0, 200, 80) if fill_level < 0.8 else (0, 60, 255)
            cv2.rectangle(frame, (20, 20), (20 + bar_total, 40), (50, 50, 50), -1)
            if bar_filled > 0:
                cv2.rectangle(frame, (20, 20), (20 + bar_filled, 40), bar_color, -1)
            cv2.rectangle(frame, (20, 20), (20 + bar_total, 40), (80, 80, 80), 1)
            cv2.putText(frame, "FATIGUE", (235, 37), cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 150, 150), 1)
            if fill_level >= 1.0:
                status_text = "DROWSY!!"
                status_color = (0, 60, 255)
                if EYE_COUNTER == FRAME_CHECK:
                    play_sound(1000, 3000)
            cv2.putText(frame, f"STATUS: {status_text}", (w // 2 - 90, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"YAWNS: {YAWN_COUNT}", (w - 160, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            if fill_level >= 1.0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (w // 2 - 230, h // 2 - 45), (w // 2 + 230, h // 2 + 45), (0, 0, 130), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                cv2.putText(frame, "WAKE UP!  PULL OVER", (w // 2 - 200, h // 2 + 12), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
            if YAWN_COUNT > 0 and YAWN_COUNT % 3 == 0:
                if not YAWN_ALERT_DONE:
                    play_sound(500, 5000)
                    YAWN_ALERT_DONE = True
                cv2.putText(frame, "3 YAWNS -- TAKE A BREAK!", (w // 2 - 175, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 220), 2)
            else:
                YAWN_ALERT_DONE = False
    else:
        cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 100, 255), 2)
    with crash_alert_lock:
        if crash_alert_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w // 2 - 260, h // 2 + 55), (w // 2 + 260, h // 2 + 110), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"!! {crash_alert_msg}", (w // 2 - 230, h // 2 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 220, 255), 2)
    cv2.imshow("ADAS -- Drowsiness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("[Drowsiness] Ended.")