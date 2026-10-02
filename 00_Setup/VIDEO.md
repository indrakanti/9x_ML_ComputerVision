# Episode 00 — ML Computer Vision Linux & PyTorch Setup

Suggested title:

**ML Computer Vision Setup on Linux — Python, PyTorch, OpenCV, ONNX & CI**

Suggested thumbnail:

**LINUX → PYTORCH → VISION**

Target duration: 24–30 minutes.

## Chapters

~~~text
00:00 Course architecture
02:00 Linux tools
05:00 Python virtual environments
08:00 Scientific/CV packages
11:00 CPU PyTorch install
14:00 NVIDIA/ROCm install strategy
17:00 Environment sanity checker
20:00 pytest + lint + type checks
23:00 ONNX / deployment path
26:00 GitHub Actions
28:00 Next: tensors
~~~

## Core message

The environment is part of the experiment.

A model result is not reproducible if we do not know the Python environment, framework version, preprocessing, device, precision, and test evidence.

## Demonstration

~~~bash
python tools/environment_check.py
pytest
ruff check .
mypy src tools
~~~

Then open the GitHub Actions run and show that the same course sanity tests execute on a clean CPU-only Linux runner.
