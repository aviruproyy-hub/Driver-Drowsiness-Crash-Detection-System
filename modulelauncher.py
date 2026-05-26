import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON = sys.executable

MODULES = [
    ("Drowsiness Detector", os.path.join(SCRIPT_DIR, "DrowsyDetect.py")),
    ("Crash Detector",      os.path.join(SCRIPT_DIR, "CrashDetect.py")),
]


def main():
    procs = []

    for name, script in MODULES:
        if not os.path.isfile(script):
            print(f"[ERROR] {name}: file not found → {script}")
            continue
        print(f"[START] {name}  ({os.path.basename(script)})")
        p = subprocess.Popen(
            [PYTHON, script],
            cwd=SCRIPT_DIR,
        )
        procs.append((name, p))

    if not procs:
        print("[ERROR] No modules to run.")
        return

    try:
        while True:
            for name, p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"[STOP]  {name} exited (code {ret})")
            # If all exited, break
            if all(p.poll() is not None for _, p in procs):
                break
            # Small sleep to avoid busy-wait
            import time
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Exited")

    finally:
        for name, p in procs:
            if p.poll() is None:
                print(f"[STOP]  Terminating {name}...")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


if __name__ == "__main__":
    main()
