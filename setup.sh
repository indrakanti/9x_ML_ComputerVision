#!/usr/bin/env bash
set -euo pipefail

MODE="cpu"
if [[ ${1:-} == "--nvidia" ]]; then
  MODE="nvidia"
elif [[ ${1:-} == "--cpu" || -z ${1:-} ]]; then
  MODE="cpu"
else
  echo "Usage: ./setup.sh [--cpu|--nvidia]"
  exit 2
fi

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run setup.sh from the repository root."
  exit 2
fi

echo "9x ML Computer Vision setup"
echo "Mode: $MODE"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed. See LINUX_SETUP.md."
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
print("Python:", sys.version.split()[0])
PY

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/onnx.txt

if [[ "$MODE" == "cpu" ]]; then
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: --nvidia requested but nvidia-smi is unavailable."
    echo "This script does not install or modify GPU drivers."
    exit 1
  fi
  echo "Detected NVIDIA driver:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1
  echo "Installing the standard PyTorch packages."
  echo "For unusual CUDA/driver combinations, use the current PyTorch selector in LINUX_SETUP.md."
  python -m pip install torch torchvision
fi

python -m pip install -e .

echo
echo "Python environment installed."
echo "Now run:"
echo "  source .venv/bin/activate"
echo "  ./course.sh doctor"
echo "  ./course.sh test"
