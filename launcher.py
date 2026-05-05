"""
=============================================================
  ADAS - Advanced Driver Assistance System
  LAUNCHER (launcher.py)
=============================================================
  Starts both modules simultaneously:
    • main.py           — Drowsiness / Yawn Detection
    • crash_detection.py — Crash / G-force Detection

  Usage:
    python launcher.py

  Both windows will open side-by-side.
  Close either window or press Q in either to stop that module.
  Ctrl-C in this terminal stops everything.
=============================================================
"""

import subprocess
import sys
import time
import signal
import os

PYTHON = sys.executable   # same interpreter that's running this file

processes = []

def shutdown(sig=None, frame=None):
    print("\n[Launcher] Shutting down all modules...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    print("=" * 56)
    print("  ADAS — Driver Safety System  |  Launcher")
    print("=" * 56)
    print("  Starting Crash Detection module...")
    p_crash = subprocess.Popen([PYTHON, "crash_detection.py"])
    processes.append(p_crash)

    # Give the crash listener a moment to bind its socket
    time.sleep(1.5)

    print("  Starting Drowsiness Detection module...")
    p_drown = subprocess.Popen([PYTHON, "main.py"])
    processes.append(p_drown)

    print("\n  Both modules running.")
    print("  Press Ctrl-C here to stop everything.\n")

    # Monitor both — exit if either crashes
    while True:
        time.sleep(1)
        if p_crash.poll() is not None:
            print("[Launcher] Crash detection module exited.")
            shutdown()
        if p_drown.poll() is not None:
            print("[Launcher] Drowsiness module exited.")
            shutdown()
