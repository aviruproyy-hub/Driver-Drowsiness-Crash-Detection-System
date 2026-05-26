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
from collections import deque

CRASH_G_THRESHOLD = 2.5
ALERT_COOLDOWN_S = 5
IMU_SAMPLE_RATE = 50
MAIN_MODULE_PORT = 65432
LOG_FILE = "crash_log.txt"
CANCEL_KEY = 'c'

WINDOW_NAME = "ADAS — Crash Detection"
WIN_W, WIN_H = 900, 560
GRAPH_HISTORY = 180
GRAPH_X, GRAPH_Y = 300, 30
GRAPH_W, GRAPH_H = 570, 200
GAUGE_X, GAUGE_Y = 300, 260
GAUGE_W, GAUGE_H = 570, 30
CONFIRM_TIMEOUT_S = 15

SAMPLE_RATE = 44100
_pygame_ok = False
try:
    import pygame
    import pygame.sndarray
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
    pygame.mixer.init()
    _freq, _size, _mix_ch = pygame.mixer.get_init()
    _pygame_ok = True
except Exception:
    pass


def _make_tone(freq, dur_s, vol=0.8):
    n = max(1, int(SAMPLE_RATE * dur_s))
    t = np.linspace(0, dur_s, n, endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * vol * 32767).astype(np.int16)
    if _pygame_ok:
        _, _, ch = pygame.mixer.get_init()
        if ch >= 2:
            wave = np.column_stack([wave, wave])
    return wave


def _play_crash_siren():
    if not _pygame_ok:
        if platform.system() == "Windows":
            import winsound
            for _ in range(3):
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
        return
    try:
        for _ in range(4):
            s1 = pygame.sndarray.make_sound(_make_tone(1600, 0.15))
            s1.play()
            time.sleep(0.18)
            s2 = pygame.sndarray.make_sound(_make_tone(1200, 0.15))
            s2.play()
            time.sleep(0.18)
    except Exception:
        pass


def _play_confirm_beep():
    if not _pygame_ok:
        return
    try:
        s = pygame.sndarray.make_sound(_make_tone(880, 0.12))
        s.play()
    except Exception:
        pass


def _play_dismiss_beep():
    if not _pygame_ok:
        return
    try:
        s = pygame.sndarray.make_sound(_make_tone(440, 0.25))
        s.play()
    except Exception:
        pass


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


def _draw_graph(frame, ax_hist, ay_hist, az_hist):
    gx, gy, gw, gh = GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H

    overlay = frame.copy()
    cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (60, 60, 60), 1)

    for i in range(1, 4):
        yy = gy + int(gh * i / 4)
        cv2.line(frame, (gx, yy), (gx + gw, yy), (40, 40, 40), 1)

    cv2.putText(frame, "Accelerometer (g)", (gx + 5, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _plot(data, color, scale=4.0):
        if len(data) < 2:
            return
        pts = []
        for i, v in enumerate(data):
            x = gx + int(i * gw / GRAPH_HISTORY)
            y = gy + gh // 2 - int((v / scale) * (gh // 2))
            y = max(gy, min(gy + gh, y))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

    _plot(list(ax_hist), (80, 180, 255), scale=5.0)
    _plot(list(ay_hist), (80, 255, 120), scale=5.0)
    _plot([v - 1.0 for v in az_hist], (255, 120, 80), scale=5.0)

    legend_items = [("X", (80, 180, 255)), ("Y", (80, 255, 120)), ("Z", (255, 120, 80))]
    lx = gx + gw - 120
    for j, (lbl, col) in enumerate(legend_items):
        cv2.line(frame, (lx + j * 40, gy + 12), (lx + j * 40 + 15, gy + 12), col, 2)
        cv2.putText(frame, lbl, (lx + j * 40 + 18, gy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)


def _draw_gauge(frame, magnitude: float, threshold: float):
    gx, gy, gw, gh = GAUGE_X, GAUGE_Y, GAUGE_W, GAUGE_H
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (30, 30, 30), -1)

    ratio = min(magnitude / (threshold * 2), 1.0)
    filled = int(ratio * gw)

    if magnitude < threshold * 0.5:
        col = (0, 180, 0)
    elif magnitude < threshold:
        t = (magnitude - threshold * 0.5) / (threshold * 0.5)
        col = (0, int(180 * (1 - t)), int(220 * t))
    else:
        col = (0, 0, 220)

    if filled > 0:
        cv2.rectangle(frame, (gx, gy), (gx + filled, gy + gh), col, -1)
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (80, 80, 80), 1)

    tx = gx + int((threshold / (threshold * 2)) * gw)
    cv2.line(frame, (tx, gy - 3), (tx, gy + gh + 3), (0, 0, 255), 2)

    cv2.putText(frame, f"Impact: {magnitude:.1f}g  (threshold: {threshold:.1f}g)",
                (gx, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)


def _draw_side_panel(frame, state: dict):
    pw = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, WIN_H), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "CRASH DETECTOR", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.line(frame, (12, 38), (pw - 12, 38), (0, 200, 255), 1)

    rows = [
        ("Mode",       state.get("mode", "SIMULATION")),
        ("Status",     state.get("status", "MONITORING")),
        ("G-Force",    f"{state.get('g_total', 0.0):.2f}"),
        ("Crashes",    str(state.get("crash_count", 0))),
        ("Dismissed",  str(state.get("dismissed", 0))),
        ("Uptime",     state.get("uptime", "00:00")),
    ]
    y0 = 68
    for label, val in rows:
        cv2.putText(frame, f"{label:<12}{val}", (12, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1)
        y0 += 24

    cv2.line(frame, (12, y0 + 4), (pw - 12, y0 + 4), (50, 50, 50), 1)
    y0 += 28
    cv2.putText(frame, "CONTROLS", (12, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 220), 1)
    y0 += 24
    controls = [
        "SPACE  Simulate crash",
        "C      Cancel alert",
        "Q/ESC  Quit",
    ]
    for c in controls:
        cv2.putText(frame, c, (12, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
        y0 += 20

    status = state.get("status", "MONITORING")
    badge_colors = {
        "MONITORING":       (0, 150, 0),
        "CRASH DETECTED":   (0, 0, 200),
        "AWAITING CONFIRM": (0, 80, 220),
        "CONFIRMED":        (0, 0, 180),
        "DISMISSED":        (150, 120, 0),
    }
    badge_col = badge_colors.get(status, (80, 80, 80))
    cv2.rectangle(frame, (12, WIN_H - 50), (pw - 12, WIN_H - 14), badge_col, -1)
    fs = 0.55 if len(status) <= 12 else 0.42
    cv2.putText(frame, status, (20, WIN_H - 26),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)


def _draw_confirmation(frame, remaining_s: float, crash_msg: str):
    cx, cy = WIN_W // 2 + 20, WIN_H // 2 + 40
    bw, bh = 420, 180

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y1 + 36), (0, 0, 180), -1)
    cv2.putText(frame, "! CRASH DETECTED !", (x1 + 95, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(frame, crash_msg, (x1 + 20, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, "Press 'C' to cancel, or alert auto-confirms:",
                (x1 + 20, y1 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    timer_col = (0, 200, 255) if remaining_s > 5 else (0, 0, 255)
    cv2.putText(frame, f"Auto-confirm in {int(remaining_s)}s",
                (x1 + 20, y1 + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, timer_col, 1)

    bar_y = y1 + 130
    bar_ratio = remaining_s / CONFIRM_TIMEOUT_S
    cv2.rectangle(frame, (x1 + 20, bar_y), (x2 - 20, bar_y + 8), (50, 50, 50), -1)
    bar_fill = int(bar_ratio * (bw - 40))
    if bar_fill > 0:
        cv2.rectangle(frame, (x1 + 20, bar_y), (x1 + 20 + bar_fill, bar_y + 8), timer_col, -1)

    btn_y = y1 + 148
    cv2.rectangle(frame, (x1 + 120, btn_y), (x1 + 300, btn_y + 28), (0, 130, 0), -1)
    cv2.putText(frame, "[C] CANCEL ALERT", (x1 + 132, btn_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)


def _draw_flash(frame, intensity: float):
    if intensity <= 0:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 200), -1)
    alpha = min(intensity, 0.5)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def main():
    imu = SimulatedIMU()
    print("[CrashDetect] SIMULATION mode. SPACE=crash, C=cancel, Q=quit")

    ax_hist = deque(maxlen=GRAPH_HISTORY)
    ay_hist = deque(maxlen=GRAPH_HISTORY)
    az_hist = deque(maxlen=GRAPH_HISTORY)

    g_history = []
    crash_detected = False
    crash_msg = ""
    last_alert_time = 0
    crash_count = 0
    alert_cancelled = False
    current_crash_id = ""
    dismissed_count = 0
    session_start = time.time()
    confirm_start = None
    flash_intensity = 0.0
    crash_confirmed = False

    state = {
        "mode":        "SIMULATION",
        "status":      "MONITORING",
        "g_total":     0.0,
        "crash_count": 0,
        "dismissed":   0,
        "uptime":      "00:00",
    }

    frame_dur = 1.0 / IMU_SAMPLE_RATE

    while True:
        t_frame = time.time()

        if crash_confirmed:
            frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            frame[:] = (15, 15, 15)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 120), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.putText(frame, "CRASH CONFIRMED",
                        (WIN_W // 2 - 240, WIN_H // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 3)
            cv2.putText(frame, crash_msg,
                        (WIN_W // 2 - 180, WIN_H // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Emergency services should be contacted.",
                        (WIN_W // 2 - 210, WIN_H // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 1)
            cv2.putText(frame, "Press Q or ESC to exit",
                        (WIN_W // 2 - 120, WIN_H // 2 + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            continue

        ax, ay, az = imu.get_accel_g()
        g_total = math.sqrt(ax**2 + ay**2 + az**2)

        ax_hist.append(ax)
        ay_hist.append(ay)
        az_hist.append(az)

        g_history.append(g_total)
        if len(g_history) > GRAPH_HISTORY:
            g_history.pop(0)

        now = time.time()

        if g_total > CRASH_G_THRESHOLD and (now - last_alert_time) > ALERT_COOLDOWN_S:
            crash_detected = True
            alert_cancelled = False
            crash_count += 1
            current_crash_id = f"CRASH_{crash_count}_{int(now)}"
            crash_msg = f"CRASH #{crash_count} | G={g_total:.2f}"
            last_alert_time = now
            confirm_start = now
            flash_intensity = 1.0
            state["status"] = "CRASH DETECTED"
            threading.Thread(target=_play_crash_siren, daemon=True).start()
            log_crash(g_total, ax, ay, az)
            send_crash_alert(crash_msg, True)

        remaining = 0
        if crash_detected and confirm_start and not alert_cancelled:
            elapsed_confirm = now - confirm_start
            remaining = max(0, CONFIRM_TIMEOUT_S - elapsed_confirm)
            state["status"] = "AWAITING CONFIRM"

            if remaining <= 0:
                crash_confirmed = True
                state["status"] = "CONFIRMED"
                print("[CrashDetect] Auto-confirmed (no response) — emergency alert triggered!")
                continue

        state["g_total"] = g_total
        state["crash_count"] = crash_count
        state["dismissed"] = dismissed_count
        state["uptime"] = time.strftime("%M:%S", time.gmtime(now - session_start))

        if not crash_detected and not alert_cancelled:
            state["status"] = "MONITORING"

        frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)

        _draw_side_panel(frame, state)
        _draw_graph(frame, ax_hist, ay_hist, az_hist)
        _draw_gauge(frame, g_total, CRASH_G_THRESHOLD)

        for i, (label, val) in enumerate([("AX", ax), ("AY", ay), ("AZ", az)]):
            col = (0, 60, 255) if abs(val) > 1.5 else (150, 150, 150)
            cv2.putText(frame, f"{label}: {val:+.3f} g", (GAUGE_X + i * 190, GAUGE_Y + GAUGE_H + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, col, 1)

        if flash_intensity > 0:
            _draw_flash(frame, flash_intensity)
            flash_intensity -= 0.03
            flash_intensity = max(0, flash_intensity)

        if crash_detected and confirm_start and not alert_cancelled:
            _draw_confirmation(frame, remaining, crash_msg)

        if alert_cancelled:
            overlay = frame.copy()
            bw, bh = 350, 80
            cx, cy = WIN_W // 2 + 20, WIN_H // 2 + 40
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 150, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, "Alert Cancelled", (x1 + 60, cy + 8),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 100), 2)
            state["status"] = "DISMISSED"
            if (now - last_alert_time) > ALERT_COOLDOWN_S:
                alert_cancelled = False
                crash_detected = False
                state["status"] = "MONITORING"
                send_crash_alert("", False)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(max(1, int(frame_dur * 1000))) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:
            break
        if key == ord(' ') and isinstance(imu, SimulatedIMU) and not crash_detected:
            imu.trigger_crash()
        if key == ord(CANCEL_KEY) and crash_detected and not alert_cancelled:
            crash_detected = False
            alert_cancelled = True
            confirm_start = None
            dismissed_count += 1
            cancel_crash_alert(current_crash_id)
            threading.Thread(target=_play_dismiss_beep, daemon=True).start()
            print(f"[CrashDetect] User dismissed — false positive, resuming monitoring.")

        elapsed_loop = time.time() - t_frame
        time.sleep(max(0, frame_dur - elapsed_loop))

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if _pygame_ok:
        pygame.mixer.quit()
    print("[CrashDetect] Stopped.")


if __name__ == "__main__":
    main()
