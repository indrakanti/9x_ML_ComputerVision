# Module 00 — Linux, Python, PyTorch & Reproducible ML Vision Setup

This module establishes the development contract for every later lesson.

## What students install

System tools:

- Git
- GCC/G++ build toolchain
- CMake
- Ninja
- pkg-config
- Python 3.10+
- Python venv/pip
- FFmpeg
- basic image runtime libraries

Python stack:

- NumPy / SciPy / pandas
- Matplotlib / Pillow
- OpenCV
- scikit-learn
- PyTorch / TorchVision
- pytest
- Ruff / mypy
- JupyterLab
- TensorBoard
- ONNX / ONNX Runtime

See the full instructions in [../LINUX_SETUP.md](../LINUX_SETUP.md).

## Course environment model

~~~text
Ubuntu/Linux
  |
Python virtual environment
  |
base scientific/CV packages
  |
PyTorch selected for CPU/CUDA/ROCm
  |
ONNX tooling
  |
editable cv9x_mlcv package
  |
environment checker
  |
pytest + lint + type checks
  |
GitHub Actions clean CPU runner
~~~

## Quick CPU setup

~~~bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/onnx.txt

python -m pip install torch torchvision   --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e .

python tools/environment_check.py
pytest
~~~

## Why PyTorch is separate

PyTorch wheel selection depends on the compute platform.

A CPU learner, an NVIDIA CUDA learner, and an AMD ROCm learner should not all be forced to install the same accelerator wheel.

The course therefore keeps general dependencies stable while PyTorch installation remains hardware-aware.

## Environment checker

Run:

~~~bash
python tools/environment_check.py
~~~

It checks:

- Python version
- NumPy
- OpenCV
- PyTorch
- TorchVision
- ONNX
- ONNX Runtime
- available ORT providers
- CUDA availability
- deterministic tensor math
- autograd gradients
- a basic OpenCV image operation

## CI philosophy

CI is CPU-first.

GPU availability is an acceleration feature, not a prerequisite for correctness.

Later GPU-specific lessons should separate:

~~~text
algorithm correctness
from
accelerator performance
~~~

## Reproducibility contract

Every training lesson should eventually identify:

- random seed
- dataset version
- preprocessing/augmentation configuration
- train/validation/test split
- model configuration
- optimizer/scheduler configuration
- checkpoint
- metrics
- software versions
- device/precision

## Next

Module 01 begins the ML foundation with NumPy arrays, tensors, shapes, broadcasting, vectorization, device movement, and dtype/precision.
