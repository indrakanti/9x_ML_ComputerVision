# Module 00 — Linux, Python/PyTorch, C++/CMake & GPU-Aware ML Vision Setup

This module establishes the development contract for every later lesson.

## What students install

System tools:

- Git
- GCC/G++ build toolchain
- CMake
- Ninja
- pkg-config
- GDB
- OpenCV C++ development package
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
                    /            \
                   /              \
          Python/PyTorch       C++17/20 + CMake
             training             runtime
                   \              /
                    \            /
                     ONNX / model contract
                            |
                         GPU path
                  CUDA / ORT / TensorRT
                            |
                profiling + monitoring
~~~

The portable CI baseline remains CPU-only, but C++ and GPU are first-class course tracks from the beginning.

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

cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
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

## C++ runtime philosophy

Python is the preferred training environment, but production lessons do not assume the Python interpreter is the final runtime.

The native path begins with C++17/20 + CMake and grows into:

~~~text
native tensor/image contracts
-> exported model
-> ONNX Runtime C++
-> CUDA execution provider
-> TensorRT
-> streaming perception runtime
~~~

## GPU philosophy

GPU is a first-class execution target.

Every major deployment checkpoint separates:

~~~text
CPU correctness
GPU correctness
GPU performance
~~~

Topics include host/device copies, pinned memory, synchronization, CUDA streams, warm-up, precision, batching, and tail latency.

## CI philosophy

GitHub Actions remains CPU-first so every PR has a portable correctness gate.

GPU CI/performance should be a separate runner/lab validation path rather than making basic correctness depend on scarce accelerator hardware.

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
- exported-model format/version
- native runtime version
- accelerator/runtime configuration

## Next

Module 01 begins with NumPy/PyTorch tensors **and the matching native runtime concepts**: shapes, HWC/CHW/NCHW, strides, dtype/precision, views/copies, CPU/GPU devices, and the C++ tensor/image memory contract.
