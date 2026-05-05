"""Crash detection module with keyboard simulation and alert system.
SPACE=crash, Q=quit, C=cancel alert."""

import cv2
import numpy as np
import threading
import socket
import json
import time
import math
import platform
import datetime
import os
USE_REAL_IMU = False
CRASH_G_THRESHOLD = 2.5
ALERT_COOLDOWN_S = 5
IMU_SAMPLE_RATE = 50
MAIN_MODULE_PORT = 65432
LOG_FILE = "crash_log.txt"
CANCEL_KEY = 'c'
def play_sound(frequency: int, duration_ms: int) -> None:
    def _beep():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(frequency, duration_ms)
        else:
            try:
                import subprocess
                subprocess.run(["beep", f"-f {frequency}", f"-l {duration_ms}"], stderr=subprocess.DEVNULL)
            except:
                print("\a", end="", flush=True)
    threading.Thread(target=_beep, daemon=True).start()
class MPU6050:
    ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    ACCEL_SCALE = 16384.0
    def __init__(self):
        import smbus2
        self.bus = smbus2.SMBus(1)
        self.bus.write_byte_data(self.ADDR, self.PWR_MGMT_1, 0)
    def _read_word_2c(self, reg: int) -> int:
        hi = self.bus.read_byte_data(self.ADDR, reg)
        lo = self.bus.read_byte_data(self.ADDR, reg + 1)
        val = (hi << 8) + lo
        return val - 65536 if val >= 0x8000 else val
    def get_accel_g(self):
        ax = self._read_word_2c(self.ACCEL_XOUT_H) / self.ACCEL_SCALE
        ay = self._read_word_2c(self.ACCEL_XOUT_H + 2) / self.ACCEL_SCALE
        az = self._read_word_2c(self.ACCEL_XOUT_H + 4) / self.ACCEL_SCALE
        return ax, ay, az
class SimulatedIMU:
    def __init__(self):
        self._crash_flag = False
        self._lock = threading.Lock()
    def trigger_crash(self):
        with self._lock:
            self._crash_flag = True
    def get_accel_g(self):
        with self._lock:
            if self._crash_flag:
                self._crash_flag = False
                return np.random.uniform(2.8, 4.5), np.random.uniform(-1.5, 1.5), np.random.uniform(-1.0, 1.0)
        return np.random.normal(0.05, 0.04), np.random.normal(0.03, 0.03), np.random.normal(1.00, 0.05)
def send_crash_alert(message: str, crashed: bool, port: int = MAIN_MODULE_PORT):
    payload = json.dumps({"crash": crashed, "message": message}).encode()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", port))
        s.sendall(payload)
        s.close()
    except:
        pass
def log_crash(g_total: float, ax: float, ay: float, az: float):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] CRASH DETECTED | G={g_total:.3f} | ax={ax:.3f}  ay={ay:.3f}  az={az:.3f}\n")
    print(f"[CrashDetect] Logged at {ts} — G={g_total:.3f}")
def cancel_crash_alert(crash_id: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] CRASH ALERT CANCELLED | ID={crash_id}\n")
    print(f"[CrashDetect] Crash alert cancelled: {crash_id}")
    payload = json.dumps({"crash": False, "message": "CRASH_ALERT_CANCELLED", "id": crash_id}).encode()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", MAIN_MODULE_PORT))
        s.sendall(payload)
        s.close()
    except:
        pass
DASH_W, DASH_H = 640, 480
HISTORY_LEN = 120
def draw_gauge(frame, cx: int, cy: int, radius: int, value: float, max_val: float, label: str, unit: str, color: tuple):
    for angle in range(210, -31, -1):
        rad = math.radians(angle)
        x = int(cx + (radius - 8) * math.cos(rad))
        y = int(cy - (radius - 8) * math.sin(rad))
        cv2.circle(frame, (x, y), 4, (50, 50, 50), -1)
    fraction = min(value / max_val, 1.0)
    sweep_end = int(210 - fraction * 240)
    for angle in range(210, sweep_end - 1, -1):
        rad = math.radians(angle)
        x = int(cx + (radius - 8) * math.cos(rad))
        y = int(cy - (radius - 8) * math.sin(rad))
        cv2.circle(frame, (x, y), 4, color, -1)
    needle_angle = math.radians(210 - fraction * 240)
    nx = int(cx + (radius - 25) * math.cos(needle_angle))
    ny = int(cy - (radius - 25) * math.sin(needle_angle))
    cv2.line(frame, (cx, cy), (nx, ny), (255, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 6, (200, 200, 200), -1)
    cv2.putText(frame, f"{value:.2f} {unit}", (cx - 45, cy + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, label, (cx - 30, cy + 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 150, 150), 1)


def draw_history_graph(frame, history: list, x: int, y: int, gw: int, gh: int, threshold: float):
    cv2.rectangle(frame, (x, y), (x + gw, y + gh), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + gw, y + gh), (70, 70, 70), 1)
    th_y = int(y + gh - (threshold / 5.0) * gh)
    cv2.line(frame, (x, th_y), (x + gw, th_y), (0, 60, 255), 1)
    cv2.putText(frame, f"threshold {threshold}g", (x + 5, th_y - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 80, 255), 1)
    if len(history) < 2:
        return
    pts = [(x + int(i * gw / HISTORY_LEN), int(y + gh - min(g / 5.0, 1.0) * gh)) for i, g in enumerate(history[-gw:])]
    for i in range(1, len(pts)):
        color = (0, 60, 255) if history[-(len(pts) - i)] > threshold else (0, 200, 80)
        cv2.line(frame, pts[i - 1], pts[i], color, 2)
    cv2.putText(frame, "G-FORCE HISTORY", (x + gw // 2 - 70, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, (130, 130, 130), 1)
def main():
    if USE_REAL_IMU:
        try:
            imu = MPU6050()
            print("[CrashDetect] MPU-6050 connected.")
        except Exception as e:
            print(f"[CrashDetect] Init failed: {e}. Using simulation.")
            imu = SimulatedIMU()
    else:
        imu = SimulatedIMU()
        print("[CrashDetect] SIMULATION mode. SPACE=crash, Q=quit, C=cancel")
    g_history = []
    crash_detected = False
    crash_msg = ""
    last_alert_time = 0
    crash_count = 0
    alert_cancelled = False
    current_crash_id = ""
    session_start = time.time()
    while True:
        loop_start = time.time()
        ax, ay, az = imu.get_accel_g()
        g_total = math.sqrt(ax**2 + ay**2 + az**2)
        g_history.append(g_total)
        if len(g_history) > HISTORY_LEN:
            g_history.pop(0)
        now = time.time()
        if g_total > CRASH_G_THRESHOLD and (now - last_alert_time) > ALERT_COOLDOWN_S:
            crash_detected = True
            alert_cancelled = False
            crash_count += 1
            current_crash_id = f"CRASH_{crash_count}_{int(now)}"
            crash_msg = f"CRASH #{crash_count} | G={g_total:.2f}"
            last_alert_time = now
            play_sound(1500, 4000)
            log_crash(g_total, ax, ay, az)
            send_crash_alert(crash_msg, True)
        elif crash_detected and not alert_cancelled and (now - last_alert_time) > ALERT_COOLDOWN_S:
            crash_detected = False
            send_crash_alert("", False)
        frame = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
        frame[:] = (15, 15, 18)
        cv2.rectangle(frame, (0, 0), (DASH_W, 65), (20, 20, 25), -1)
        cv2.line(frame, (0, 65), (DASH_W, 65), (0, 160, 80), 2)
        cv2.putText(frame, "ADAS v2.0  —  CRASH DETECTION", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1)
        elapsed = int(now - session_start)
        m, s = divmod(elapsed, 60)
        cv2.putText(frame, f"SESSION {m:02d}:{s:02d}", (DASH_W - 140, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100), 1)
        g_color = (0, 60, 255) if g_total > CRASH_G_THRESHOLD else (0, 200, 80)
        draw_gauge(frame, 110, 210, 90, g_total, 5.0, "TOTAL G", "g", g_color)
        draw_gauge(frame, 320, 210, 90, abs(ax), 3.0, "LATERAL", "g", (0, 180, 255))
        draw_gauge(frame, 530, 210, 90, abs(ay), 3.0, "VERTICAL", "g", (200, 160, 0))
        for i, (label, val) in enumerate([("AX", ax), ("AY", ay), ("AZ", az)]):
            col = (0, 60, 255) if abs(val) > 1.5 else (150, 150, 150)
            cv2.putText(frame, f"{label}: {val:+.3f} g", (20 + i * 200, 315), cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 1)
        draw_history_graph(frame, g_history, 20, 335, DASH_W - 40, 95, CRASH_G_THRESHOLD)
        cv2.rectangle(frame, (0, DASH_H - 55), (DASH_W, DASH_H), (20, 20, 25), -1)
        cv2.line(frame, (0, DASH_H - 55), (DASH_W, DASH_H - 55), (50, 50, 55), 1)
        cv2.putText(frame, f"Crashes: {crash_count}", (20, DASH_H - 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (180, 180, 180), 1)
        mode_txt = "REAL IMU" if USE_REAL_IMU else "SIM MODE"
        cv2.putText(frame, mode_txt, (DASH_W - 200, DASH_H - 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 80, 80), 1)

        if crash_detected and not alert_cancelled:
            overlay = frame.copy()
            cv2.rectangle(overlay, (DASH_W // 2 - 260, DASH_H // 2 - 70), (DASH_W // 2 + 260, DASH_H // 2 + 70), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, "⚠  CRASH DETECTED!", (DASH_W // 2 - 170, DASH_H // 2 - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 220, 255), 2)
            cv2.putText(frame, crash_msg, (DASH_W // 2 - 140, DASH_H // 2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(frame, f"Press '{CANCEL_KEY.upper()}' to cancel", (DASH_W // 2 - 150, DASH_H // 2 + 55), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 220, 255), 1)
        elif alert_cancelled:
            overlay = frame.copy()
            cv2.rectangle(overlay, (DASH_W // 2 - 200, DASH_H // 2 - 40), (DASH_W // 2 + 200, DASH_H // 2 + 40), (0, 150, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, "✓ Alert Cancelled", (DASH_W // 2 - 130, DASH_H // 2 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 100), 2)
        cv2.imshow("ADAS — Crash Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and isinstance(imu, SimulatedIMU):
            imu.trigger_crash()
        if key == ord(CANCEL_KEY) and crash_detected:
            crash_detected = False
            alert_cancelled = True
            cancel_crash_alert(current_crash_id)
            play_sound(800, 500)
        elapsed_loop = time.time() - loop_start
        time.sleep(max(0, (1.0 / IMU_SAMPLE_RATE) - elapsed_loop))

    cv2.destroyAllWindows()
    print("[CrashDetect] Stopped.")


if __name__=="__main__":
    main()
