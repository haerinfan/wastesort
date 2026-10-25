"""Standard-library bootstrap; run with Python 3.12 from any directory."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT, reinstall=False):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12: py -3.12 scripts/launch.py")
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print("Creating WasteSort's Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = root / "requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment / "wastesort-requirements.sha256"
    if reinstall or not marker.exists() or marker.read_text().strip() != fingerprint:
        print("Installing dependencies. The first setup may take several minutes...", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)],
                       cwd=root, check=True)
        # Only mark successful installs. A failed installation can be retried.
        marker.write_text(fingerprint, encoding="ascii")
    return python


def main():
    parser = argparse.ArgumentParser(description="Set up and launch WasteSort")
    parser.add_argument("--check", action="store_true", help="Check installation without opening the app")
    parser.add_argument("--reinstall", action="store_true", help="Recheck and install pinned dependencies")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        python = prepare(reinstall=args.reinstall)
        subprocess.run([str(python), "-m", "wastesort.doctor"], cwd=ROOT, check=True)
        if not args.check:
            print(f"Open http://localhost:{args.port}. Keep this window open; Ctrl+C stops the app.", flush=True)
            subprocess.run([str(python), "-m", "streamlit", "run", "app.py",
                            "--server.address=127.0.0.1", f"--server.port={args.port}",
                            "--browser.gatherUsageStats=false"], cwd=ROOT, check=True)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup stopped: {exc}\nSee docs/RUNNING.md for help.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
