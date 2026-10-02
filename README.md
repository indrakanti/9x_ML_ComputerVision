# 9x ML Computer Vision

**From machine-learning foundations to modern multimodal and embodied vision — Python/PyTorch first, deployment-aware, production-minded.**

This repository is the learned/deep-vision continuation of the classical C++ course in `9x_ComputerVision_-`.

## Course identity

The course deliberately separates two concerns:

1. **Learn the model and training system in Python/PyTorch.**
2. **Understand how the trained model is evaluated, exported, optimized, and deployed.**

The default learning path is:

~~~text
Python fundamentals for ML
-> tensors + autograd
-> datasets + augmentation
-> linear/MLP classifiers
-> convolution from first principles
-> CNNs
-> transfer learning
-> modern backbones
-> detection
-> segmentation
-> pose/keypoints
-> self-supervised learning
-> vision transformers
-> multimodal / vision-language models
-> open-vocabulary + grounding
-> video understanding
-> 3-D / neural rendering
-> generative vision
-> world / embodied models
-> vision-language-action
-> ONNX / optimized inference / C++ deployment
~~~

## Design principles

- Python/PyTorch is the primary teaching environment.
- Mathematical ideas are explained before framework convenience APIs.
- Notebooks are used for visualization/exploration, not as the only implementation.
- Reusable code lives in importable Python modules.
- Every module should include deterministic tests where practical.
- CPU execution remains a supported baseline.
- GPU acceleration is optional and hardware-aware.
- Training, validation, and test splits remain explicit.
- Reproducibility includes seeds, environment, preprocessing, checkpoints, and metrics.
- Model export/deployment is part of the curriculum, not an afterthought.
- Modern concepts are added by capability category rather than chasing every transient model release.

## Linux quick start

See [LINUX_SETUP.md](LINUX_SETUP.md).

Minimal CPU path:

~~~bash
sudo apt update
sudo apt install -y \
  git build-essential cmake ninja-build pkg-config \
  python3 python3-dev python3-pip python3-venv \
  curl wget unzip ffmpeg libgl1 libglib2.0-0

git clone https://github.com/indrakanti/9x_ML_ComputerVision.git
cd 9x_ML_ComputerVision

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/onnx.txt

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python tools/environment_check.py
pytest
~~~

For NVIDIA/AMD GPU installs, use the current PyTorch platform selector rather than assuming one CUDA/ROCm wheel fits every system.

## Repository structure

~~~text
9x_ML_ComputerVision/
├── 00_Setup/
├── src/cv9x_mlcv/
├── tests/
├── tools/
├── requirements/
├── COURSE_ROADMAP.md
├── VIDEO_SERIES.md
├── LINUX_SETUP.md
├── pyproject.toml
└── .github/workflows/
~~~

Future course modules will be added as focused pull requests.

## Course roadmap

See [COURSE_ROADMAP.md](COURSE_ROADMAP.md).

The roadmap intentionally reaches beyond traditional CNN/YOLO material into:

- self-supervised vision foundation models
- multimodal vision-language models
- open-vocabulary recognition and grounding
- promptable/foundation segmentation
- video representation and temporal reasoning
- 3-D neural scene representations
- generative/diffusion/flow-based vision
- world models
- embodied perception
- vision-language-action systems

## Relationship to the classical course

The classical repository teaches:

~~~text
pixels -> filters -> features -> geometry -> stereo -> motion
-> classical recognition -> tracking -> production-oriented C++
~~~

This repository continues with:

~~~text
learned representation -> training -> modern vision models
-> multimodal reasoning -> embodied vision -> deployment
~~~

The two courses are complementary rather than duplicates.
