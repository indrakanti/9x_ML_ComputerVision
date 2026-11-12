# 9x ML Computer Vision

**From ML foundations to modern multimodal and embodied vision — Python/PyTorch for training, C++ for production runtime, GPU-aware from day one.**

This repository is the learned/deep-vision continuation of the classical C++ course in `9x_ComputerVision_-`.

## Course series and completion boundaries

The 86 planned modules are intentionally split into three independently completable courses:

| Course | Modules | Student outcome |
|---|---:|---|
| **ML Computer Vision — Part 1: Foundations to Modern Perception** | 00–31 | Train, evaluate, export and deploy modern image models |
| **ML Computer Vision — Part 2: Foundation, Multimodal, Video & Generative Vision** | 32–60 | Work with VLMs, open-vocabulary vision, video, learned 3-D and generative vision |
| **ML Computer Vision — Part 3: Embodied AI & Production Vision Systems** | 61–85 | Understand world/VLA models and build production inference/runtime systems |

Each part has a defined completion point. Learners do not need to finish all 86 modules before gaining a complete, useful skill set.

## Course identity

The course uses three coordinated tracks:

1. **Learning / training:** Python + PyTorch.
2. **Production runtime:** C++17 baseline + CMake + native inference runtimes. Optional examples may use C++20 only when explicitly enabled.
3. **Acceleration:** CUDA/GPU execution, profiling, memory movement, synchronization, and optimized inference.

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
-> ONNX / optimized inference
-> C++ runtime integration
-> CUDA / TensorRT / edge deployment
~~~

## Design principles

- Python/PyTorch is the primary training and experimentation environment.
- C++ is a first-class runtime/deployment language from Module 00 onward.
- GPU execution is a first-class systems topic, not an end-of-course appendix.
- Mathematical ideas are explained before framework convenience APIs.
- Notebooks are used for visualization/exploration, not as the only implementation.
- Reusable Python code lives in importable modules; reusable native code lives under the C++ track.
- Every module should include deterministic tests where practical.
- CPU execution remains the portable correctness baseline.
- GPU correctness and GPU performance are validated separately.
- Training, validation, and test splits remain explicit.
- Reproducibility includes seeds, environment, preprocessing, checkpoints, metrics, runtime version, device, and precision.
- Model export/deployment is integrated throughout the curriculum.
- Safety-oriented runtime topics include ownership, bounded interfaces, lifecycle, monitoring, timing, and fault handling.
- Modern concepts are added by capability category rather than chasing every transient model release.

## Linux quick start

See [LINUX_SETUP.md](LINUX_SETUP.md).

Beginner CPU path:

~~~bash
sudo apt update
sudo apt install -y \
  git build-essential cmake ninja-build pkg-config \
  python3 python3-dev python3-pip python3-venv \
  curl wget unzip ffmpeg libgl1 libglib2.0-0 libopencv-dev

git clone https://github.com/indrakanti/9x_ML_ComputerVision.git
cd 9x_ML_ComputerVision

bash setup.sh --cpu
source .venv/bin/activate
bash course.sh doctor
~~~

A learner with no GPU is fully supported. For an NVIDIA machine, use `bash setup.sh --nvidia`; the script detects the driver but never modifies system GPU drivers or installs a CUDA toolkit behind the user's back.

After `course.sh doctor` reports **COURSE ENVIRONMENT READY**, the workstation is ready for Module 01.

For NVIDIA/AMD GPU installs, use the current PyTorch platform selector rather than assuming one CUDA/ROCm wheel fits every system.

## Repository structure

~~~text
9x_ML_ComputerVision/
├── 00_Setup/
├── cpp/
├── src/cv9x_mlcv/
├── tests/
├── tools/
├── requirements/
├── CMakeLists.txt
├── COURSE_ROADMAP.md
├── VIDEO_SERIES.md
├── LINUX_SETUP.md
├── pyproject.toml
└── .github/workflows/
~~~

Future course modules will be added as focused pull requests.

## Course resources

- [Course roadmap](COURSE_ROADMAP.md)
- [Linux setup](LINUX_SETUP.md)
- [Runtime architecture](RUNTIME_ARCHITECTURE.md)
- [Standard module template](MODULE_TEMPLATE.md)
- [Contributing](CONTRIBUTING.md)
- [Apache License 2.0](LICENSE)

The standard major-model lifecycle is:

~~~text
PyTorch reference
      ↓
training + evaluation
      ↓
exported artifact
      ↓
ONNX Runtime Python
      ↓
ONNX Runtime C++
      ↓
GPU / optimized runtime
      ↓
profiling + monitoring
~~~

Each transition should be verified, not assumed.

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
learned representation
-> Python/PyTorch training
-> modern vision models
-> multimodal / embodied vision
-> ONNX export
-> C++ runtime
-> GPU acceleration
-> safety-oriented production integration
~~~

The two courses are complementary rather than duplicates.

See [RUNTIME_ARCHITECTURE.md](RUNTIME_ARCHITECTURE.md) for the production runtime contract.

## Runtime philosophy

A model is not considered "production-ready" merely because it produces correct outputs in Python.

Major course milestones will connect trained models to native execution and explicitly examine:

- tensor/image layout contracts
- host/device copies
- pinned memory
- asynchronous execution
- CUDA streams and synchronization
- warm-up
- batching vs streaming
- FP32 / FP16 / BF16 / INT8
- ONNX Runtime execution providers
- TensorRT engines/contexts
- latency distributions and tail latency
- memory use
- deterministic startup/shutdown
- model/version compatibility
- runtime error handling
- monitoring and fallback behavior

This is especially important for automotive, robotics, and other safety-oriented systems.
