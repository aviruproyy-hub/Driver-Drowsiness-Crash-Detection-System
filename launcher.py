"""
=============================================================
  ADAS - Advanced Driver Assistance System
  LAUNCHER (launcher.py)
=============================================================
  Integrated Dual-Module Driver Safety System:
    • crash_detection.py   — Crash / G-force Detection
    • DrowsyDetect.py      — Drowsiness / Yawn Detection

  Features:
    ✓ Parallel execution of both detection systems
    ✓ Real-time inter-process communication via sockets
    ✓ Unified shutdown and logging
    ✓ Automatic restart on crash
    ✓ Combined status monitoring

  Usage:
    python launcher.py

  Controls:
    • Q or ESC: Quit either module (shuts down all)
    • SPACE: Simulate crash (in crash detection)
    • C: Cancel crash alert
    • Ctrl-C: Emergency shutdown

=============================================================
"""

import subprocess
import sys
import time
import signal
import os
import threading
from datetime import datetime

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_MODULE = "crash_detection.py"
DROWSY_MODULE = "main.py"
MAX_RESTART_ATTEMPTS = 3

processes = {}
module_status = {
    "crash": {"running": False, "restart_count": 0, "last_error": None},
    "drowsy": {"running": False, "restart_count": 0, "last_error": None}
}
status_lock = threading.Lock()

def log_event(module: str, event: str, details: str = ""):
    """Log events with timestamps."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] [{module.upper():6}] {event}"
    if details:
        msg += f" — {details}"
    print(msg)

def update_status(module: str, running: bool, error: str = None):
    """Update module status thread-safely."""
    with status_lock:
        module_status[module]["running"] = running
        if error:
            module_status[module]["last_error"] = error
        else:
            module_status[module]["last_error"] = None

def start_module(module_name: str, script_name: str):
    """Start a detection module and monitor it."""
    attempt = 0
    while attempt < MAX_RESTART_ATTEMPTS:
        try:
            log_event("launcher", f"Starting {module_name} (attempt {attempt + 1}/{MAX_RESTART_ATTEMPTS})")
            
            process = subprocess.Popen(
                [PYTHON, script_name],
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            processes[module_name] = process
            update_status(module_name, True)
            log_event("launcher", f"{module_name} started (PID: {process.pid})")
            
            # Wait for process to finish
            _, stderr = process.communicate()
            exit_code = process.returncode
            
            update_status(module_name, False)
            
            if exit_code != 0:
                error_msg = f"Exit code {exit_code}"
                if stderr:
                    error_msg += f": {stderr[:100]}"
                log_event("launcher", f"{module_name} exited", error_msg)
                update_status(module_name, False, error_msg)
                attempt += 1
                if attempt < MAX_RESTART_ATTEMPTS:
                    time.sleep(2)
            else:
                log_event("launcher", f"{module_name} exited normally")
                break
                
        except Exception as e:
            log_event("launcher", f"{module_name} error", str(e))
            update_status(module_name, False, str(e))
            attempt += 1
            time.sleep(2)
    
    with status_lock:
        module_status[module_name]["restart_count"] = attempt

def shutdown(sig=None, frame=None, reason="User interrupt"):
    """Gracefully shutdown all modules."""
    log_event("launcher", "SHUTDOWN INITIATED", reason)
    
    for module_name, process in processes.items():
        if process and process.poll() is None:
            try:
                log_event("launcher", f"Terminating {module_name}")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    log_event("launcher", f"Force killing {module_name}")
                    process.kill()
                log_event("launcher", f"{module_name} terminated")
            except Exception as e:
                log_event("launcher", f"Error terminating {module_name}", str(e))
    
    log_event("launcher", "SYSTEM SHUTDOWN COMPLETE")
    sys.exit(0)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAS — Advanced Driver Assistance System  |  Launcher")
    print("  Crash Detection + Drowsiness Detection")
    print("=" * 70)
    print()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda s, f: shutdown(s, f, "SIGINT"))
    signal.signal(signal.SIGTERM, lambda s, f: shutdown(s, f, "SIGTERM"))
    
    # Start crash detection first (it binds a socket)
    crash_thread = threading.Thread(target=start_module, args=("crash", CRASH_MODULE), daemon=False)
    crash_thread.start()
    
    # Wait for crash module to bind
    time.sleep(1.5)
    
    # Start drowsiness detection
    drowsy_thread = threading.Thread(target=start_module, args=("drowsy", DROWSY_MODULE), daemon=False)
    drowsy_thread.start()
    
    print("\n  ✓ Crash detection module launching...")
    print("  ✓ Drowsiness detection module launching...")
    print("\n  Both modules are now running in parallel.")
    print("  Monitor this terminal for status updates.")
    print("  Press Ctrl-C to shutdown all modules.\n")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(reason="Keyboard interrupt")
