"""Run with native Python 3.12; installs only into this standalone project."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    venv = root / '.venv'
    python = venv / 'bin/python'
    if not (3,12) <= sys.version_info[:2] < (3,15):
        raise SystemExit('Use Python 3.12-3.14; the measured reference uses 3.12.1.')
    if not python.exists(): subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    # uv-created environments may omit pip; ensurepip uses Python's bundled wheel.
    has_pip = subprocess.run([str(python), '-c', 'import pip'], capture_output=True).returncode == 0
    if not has_pip: subprocess.run([str(python), '-m', 'ensurepip'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(root/'requirements.lock')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '-e', str(root)], check=True)
    subprocess.run([str(python), '-m', 'pytest', str(root/'tests')], cwd=str(root), check=True)


if __name__ == '__main__': main()
