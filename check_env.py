"""
Environment smoke test.

Run this once on your machine before committing GPU time:

    python check_env.py

It checks the Python version, the torch / CUDA install (the cu128 wheel trap that
silently falls back to CPU on RTX 50-series cards), that the torch trio versions
match, that ffmpeg is on PATH, and that the core packages import.
"""

import importlib
import shutil
import sys

OK = "[ ok ]"
WARN = "[warn]"
FAIL = "[fail]"

problems = []
warnings = []


def line(status, msg):
    print(f"{status} {msg}")


def check_python():
    v = sys.version_info
    msg = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        line(OK, msg)
    elif v.major == 3 and 9 <= v.minor <= 12:
        line(WARN, msg + " (3.11 is recommended)")
        warnings.append("Python is not 3.11")
    else:
        line(FAIL, msg + " (need Python 3.9 to 3.12, 3.11 recommended)")
        problems.append("Python version")


def check_torch():
    try:
        import torch
    except ImportError:
        line(FAIL, "torch is not installed. Install from the cu128 wheel index (see README).")
        problems.append("torch missing")
        return

    line(OK, f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        line(WARN, "CUDA not available to torch. Training will run on CPU and be very slow.")
        line("      ", "If you have an NVIDIA GPU, you likely installed the default PyPI wheel.")
        line("      ", "Reinstall from --index-url https://download.pytorch.org/whl/cu128")
        warnings.append("CUDA not available")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]
    line(OK, f"CUDA device: {name} (sm_{sm})")

    # Blackwell (RTX 50-series) is sm_120 and needs the cu128 wheel.
    if sm >= 120:
        line("      ", "Blackwell class GPU detected. Confirm this is the cu128 build.")

    # A real allocation confirms kernels actually run on this card.
    try:
        x = torch.randn(8, 8, device="cuda")
        _ = (x @ x).sum().item()
        line(OK, "CUDA tensor op succeeded")
    except Exception as exc:
        line(FAIL, f"CUDA tensor op failed: {exc}")
        line("      ", "This is the sm_120 mismatch symptom. Reinstall torch from the cu128 index.")
        problems.append("CUDA op failed")


def check_trio():
    try:
        import torchvision
        import torchaudio
    except ImportError:
        return  # torch failure already reported above
    tv = torchvision.__version__.split("+")[0]
    ta = torchaudio.__version__.split("+")[0]
    line(OK, f"torchvision {tv}, torchaudio {ta}")
    # The trio must share a release line or you get symbol errors at import time.


def check_ffmpeg():
    if shutil.which("ffmpeg"):
        line(OK, "ffmpeg found on PATH")
    else:
        line(FAIL, "ffmpeg not found on PATH. Needed for video decoding.")
        problems.append("ffmpeg missing")


def check_packages():
    # import name -> pip name, for anything where they differ
    pkgs = {
        "numpy": "numpy",
        "pandas": "pandas",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "timm": "timm",
        "torchmetrics": "torchmetrics",
        "yaml": "pyyaml",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
    }
    missing = []
    for import_name, pip_name in pkgs.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        line(OK, f"core packages import ({len(pkgs)} checked)")
    else:
        line(WARN, "missing packages: " + ", ".join(missing))
        line("      ", "Install with: pip install -r requirements.txt")
        warnings.append("packages missing")


def main():
    print("Environment smoke test")
    print("=" * 44)
    check_python()
    check_torch()
    check_trio()
    check_ffmpeg()
    check_packages()
    print("=" * 44)
    if problems:
        print(f"{FAIL} {len(problems)} blocking issue(s): " + ", ".join(problems))
        return 1
    if warnings:
        print(f"{WARN} {len(warnings)} warning(s), nothing blocking.")
        return 0
    print(f"{OK} all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
