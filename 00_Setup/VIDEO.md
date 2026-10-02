# Episode 00 — ML Computer Vision Linux & PyTorch Setup

Suggested title:

**ML Computer Vision Setup — Python/PyTorch + C++ + GPU on Linux**

Suggested thumbnail:

**PYTHON TRAIN. C++ RUN. GPU ACCELERATE.**

Target duration: 24–30 minutes.

## Chapters

~~~text
00:00 Course architecture
02:00 Linux tools
05:00 Python virtual environments
08:00 Scientific/CV packages
11:00 CPU PyTorch install
14:00 NVIDIA/ROCm install strategy
17:00 C++/CMake native runtime
20:00 CUDA toolkit vs PyTorch CUDA runtime
23:00 Environment + native sanity checks
26:00 ONNX Runtime / TensorRT direction
29:00 CPU CI vs GPU validation
32:00 Next: tensors
~~~

## Core message

The environment is part of the experiment.

A model result is not production-ready if we only know how to run it from a Python training script. The course tracks the Python training environment, native C++ runtime, model contract, accelerator, precision, and test evidence.

## Demonstration

~~~bash
python tools/environment_check.py
pytest
ruff check .
mypy src tools

cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
~~~

Then open the GitHub Actions run and show that the same course sanity tests execute on a clean CPU-only Linux runner.
