# Linux Setup — 9x ML Computer Vision

This guide targets Ubuntu/Debian-style Linux systems.

## 1. Base operating-system tools

~~~bash
sudo apt update

sudo apt install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  curl \
  wget \
  unzip \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libjpeg-dev \
  libpng-dev \
  libopencv-dev \
  gdb
~~~

Useful optional tools:

~~~bash
sudo apt install -y htop tree
~~~

## 2. Verify the native toolchain

~~~bash
git --version
g++ --version
cmake --version
ninja --version
python3 --version
ffmpeg -version
~~~

The course requires Python 3.10 or newer.

## 3. Clone and create a virtual environment

~~~bash
git clone https://github.com/indrakanti/9x_ML_ComputerVision.git
cd 9x_ML_ComputerVision

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
~~~

Every new terminal:

~~~bash
cd 9x_ML_ComputerVision
source .venv/bin/activate
~~~

## 4. Install course Python packages

~~~bash
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/onnx.txt
~~~

PyTorch is intentionally installed separately because the correct wheel depends on the compute platform.

## 5. PyTorch — CPU-only path

For a machine without an NVIDIA/AMD accelerator, or when learning on CPU:

~~~bash
python -m pip install \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
~~~

## 6. PyTorch — NVIDIA GPU path

First inspect the system:

~~~bash
nvidia-smi
~~~

Then use the **current PyTorch Start Locally selector** for:

~~~text
OS: Linux
Package: Pip
Language: Python
Compute Platform: the CUDA option compatible with the machine
~~~

Do not copy an old CUDA wheel command from a blog or previous semester.

The PyTorch package normally brings the user-space CUDA runtime components it expects; a functioning compatible NVIDIA driver is still required.

Verify:

~~~bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
~~~

## 7. AMD ROCm path

On supported AMD GPUs, use the current PyTorch selector and the ROCm package appropriate for the installed platform.

ROCm support is hardware/OS specific; verify support before changing the course environment.

## 8. Jupyter

Jupyter is installed through requirements/dev.txt.

Start:

~~~bash
jupyter lab
~~~

Optional kernel registration:

~~~bash
python -m ipykernel install \
  --user \
  --name 9x-mlcv \
  --display-name "9x ML Computer Vision"
~~~

## 9. ONNX and ONNX Runtime

CPU:

~~~bash
python -m pip install -r requirements/onnx.txt
~~~

The default course requirement uses CPU ONNX Runtime for portable CI.

For a GPU environment, do not install both CPU and GPU ONNX Runtime variants blindly in the same environment. Follow the current ONNX Runtime compatibility guidance and replace the CPU package with the appropriate GPU package.

## 10. Native C++ sanity check

The C++ runtime path is available from Module 00.

~~~bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
~~~

Direct run:

~~~bash
./build-native/cv9x_native_sanity --self-test
~~~

Strict warnings:

~~~bash
cmake -S . -B build-native-werror \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-native-werror --parallel
~~~

## 11. CUDA toolkit / nvcc path

PyTorch GPU execution does not automatically require a system-wide CUDA toolkit, but native CUDA development and custom CUDA code do.

If a CUDA toolkit is installed:

~~~bash
nvcc --version
~~~

Probe it through the course build:

~~~bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_ENABLE_CUDA=ON
~~~

This fails intentionally when CUDA is requested but no CUDA compiler is present.

Later lessons will add real CUDA, ONNX Runtime GPU, and TensorRT C++ targets behind explicit build/runtime checks.

## 12. Environment sanity check

~~~bash
python tools/environment_check.py
~~~

Expected categories:

~~~text
Python
NumPy
OpenCV
PyTorch
TorchVision
ONNX
ONNX Runtime
device availability
tensor math
autograd
OpenCV operation
~~~

## 13. Run Python tests

~~~bash
pytest
~~~

Coverage:

~~~bash
pytest --cov=cv9x_mlcv --cov-report=term-missing
~~~

## 14. Lint / static checks

~~~bash
ruff check .
mypy src tools
~~~

## 15. CPU-first CI policy

GitHub Actions runs on CPU by default.

Why:

- every contributor can reproduce it
- no accelerator quota is required
- correctness tests should not depend on CUDA
- GPU performance is tested separately from algorithm correctness

GPU-specific modules should still provide CPU smoke paths where practical.

## 16. CUDA terminology

Keep these separate:

- **NVIDIA driver** — kernel/user driver required to access the GPU
- **CUDA runtime/toolkit** — libraries/compiler ecosystem
- **PyTorch CUDA build** — PyTorch wheel compiled for a supported CUDA runtime family

A system-wide CUDA toolkit is not automatically required just to run a prebuilt PyTorch CUDA wheel.

It becomes relevant when compiling custom CUDA code/extensions or using tools that require nvcc.

## 17. Native inference dependencies

The base C++ compiler/CMake/OpenCV/debugger toolchain is installed on day one.

Large native ML runtimes remain lesson-scoped so students do not need every deployment SDK immediately:

- LibTorch when teaching native PyTorch tensor/module execution
- ONNX Runtime C++ for portable exported-model inference
- CUDA execution providers on supported NVIDIA systems
- TensorRT for NVIDIA-optimized deployment

This keeps the initial environment reproducible while making native execution a first-class course path.

## 18. Clean environment reset

~~~bash
deactivate 2>/dev/null || true
rm -rf .venv

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/onnx.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~~~

## 19. Common failures

### Python too old

Check:

~~~bash
python3 --version
~~~

Use Python 3.10+.

### torch imports but CUDA is unavailable

Check:

~~~bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
~~~

Verify that the installed PyTorch wheel matches the intended compute platform and that the driver is functioning.

### OpenCV import fails with shared-library errors

Ensure the Linux runtime packages such as libgl1 and libglib2.0-0 are installed.

### Jupyter uses the wrong Python

Inside the notebook:

~~~python
import sys
print(sys.executable)
~~~

It should point to the intended environment.

### ONNX Runtime package conflict

Keep one appropriate runtime variant per environment unless official compatibility guidance explicitly supports another setup.
