# Part I Review — Foundations to Modern Vision

## Scope

Part I covers Modules **00–31**: from ML foundations through modern image perception, self-supervised representation learning, parameter-efficient adaptation, and compression.

The completion boundary is intentionally:

```text
understand the math
        ↓
build deterministic references
        ↓
train/evaluate learned models
        ↓
export a real model
        ↓
verify native numerical contracts
        ↓
introduce GPU/runtime constraints
        ↓
adapt and compress representations
```

It does **not** claim to finish the full production inference stack. Full ONNX Runtime C++, TensorRT, edge profiling, and production monitoring are dedicated Part-3 modules.

## What Stage 1 now contains

### Foundations — Modules 00–05

- reproducible Linux/Python/C++ environment
- tensors and memory contracts
- probability/statistics
- optimization
- autograd
- dataset splits, leakage, metrics, reproducibility

### Learned features and CNN training — Modules 06–11

- linear classifiers
- MLPs
- convolution from first principles
- CNN training
- regularization/augmentation/schedulers/checkpoints
- transfer learning
- first real ONNX export and ONNX Runtime parity

### Modern backbones — Modules 12–17

- residual learning
- depthwise-separable/efficient CNNs
- Vision Transformer attention
- windowed transformers
- CNN–Transformer hybrids
- engineering trade-space evaluation

### Detection, segmentation, and pose — Modules 18–25

- boxes, IoU, NMS, AP/mAP
- YOLO-style one-stage decode
- optional native CUDA decode checkpoint
- two-stage proposals
- DETR-style bipartite matching
- semantic segmentation
- instance/panoptic segmentation
- promptable segmentation runtime semantics
- heatmap/keypoint pose decoding

### Representation learning and foundation vision — Modules 26–31

- CLIP-style contrastive alignment
- masked image modeling
- DINO-style self-distillation
- reusable foundation-feature contracts
- probing, adapters, LoRA/PEFT
- distillation, pruning, INT8 quantization

## Native engineering coverage

Stage 1 does not treat C++ as a final appendix.

Meaningful native counterparts exist across the course for concepts including:

- tensor/memory contracts
- statistics/optimization/autodiff
- classifier and MLP inference
- convolution
- CNN inference
- residual and efficient convolution blocks
- attention/windowed attention
- hybrid CNN–Transformer execution
- detection geometry and post-processing
- DETR matching
- segmentation metrics/composition
- promptable-mask selection
- pose decode
- embedding similarity
- masked reconstruction
- DINO math
- foundation adapters
- PEFT math
- compression math

The portable baseline remains **C++17 + CMake + CTest**.

## Numerical-equivalence policy

Where practical, the course verifies complete numerical contracts rather than only final labels.

Examples include:

```text
Python ↔ C++
PyTorch ↔ manual reference
PyTorch ↔ ONNX Runtime
PyTorch ↔ ONNX Runtime ↔ C++ reference
```

The root preflight gate runs parity scripts after the native build.

## GPU-awareness checkpoint

Stage 1 introduces GPU concepts without making CPU CI depend on a GPU.

Covered concepts include:

- host/device memory movement
- synchronization
- kernel-launch boundaries
- GPU-resident detector post-processing
- optional CUDA compilation
- timing pitfalls for asynchronous execution
- multi-camera multiplication of per-frame cost

The full optimized GPU runtime is intentionally deferred.

## Export/deployment checkpoint

Module 11 establishes the first real learned-model export path:

```text
PyTorch model
      ↓
ONNX
      ↓
ONNX graph validation
      ↓
ONNX Runtime Python
      ↓
numerical equivalence
```

Later Part-3 modules will extend this into:

```text
ONNX Runtime C++
TensorRT
quantized runtime
streaming inference
edge profiling
monitoring / drift
```

## Quality gates

Part-I changes are expected to pass:

```text
Ruff
mypy
pytest
CPU PyTorch correctness
CMake configure
C++17 build
warnings-as-errors
CTest
Python ↔ C++ parity
ONNX checks where applicable
```

The single repository command is:

```bash
./preflight.sh
```

## Book readiness

Every substantial module is intended to provide material that can be reused in the book with minimal restructuring:

```text
WHY
↓
MATH / INTUITION
↓
REFERENCE IMPLEMENTATION
↓
PYTORCH
↓
C++ NATIVE CONTRACT
↓
PARITY
↓
DEPLOYMENT IMPLICATIONS
↓
FAILURE MODES
↓
EXERCISES
↓
BOOK TAKEAWAY
```

Not every module needs every layer equally—for example, a pretraining-only concept may have no meaningful ONNX artifact—but omissions should be intentional rather than accidental.

## What a learner should be able to do after Stage 1

A learner completing Part 1 should be able to:

- reason about ML optimization and tensor computation
- train and evaluate CNN/Transformer image models
- understand modern backbone architecture choices
- implement detector post-processing and evaluation
- understand semantic, instance, panoptic, promptable, and keypoint outputs
- explain CLIP, masked modeling, DINO, and reusable visual backbones
- adapt frozen models with probes/adapters/LoRA
- reason about distillation, pruning, and quantization
- define and validate Python/native numerical contracts
- understand where memory movement and synchronization enter GPU deployment
- export a learned PyTorch model to ONNX and verify runtime equivalence

## What Stage 1 intentionally does not claim

Part 1 is **not** a claim of production-certifiable inference infrastructure.

Still ahead:

- full C++ ONNX Runtime session integration
- execution-provider management
- TensorRT engine lifecycle
- production CUDA streams/buffer pools
- mixed-precision runtime qualification
- streaming queues and backpressure
- model lifecycle / rollback
- edge-device profiling
- runtime health monitoring
- drift detection
- safety-case evidence

Those are explicit later-course topics, not hidden omissions.

## Part-I completion checkpoint

When Modules 00–31 are merged and root CI is green:

> **ML Computer Vision — Part 1: Foundations to Modern Perception is implementation-complete.**

The next module is **32 — Text/Image Embeddings and Zero-Shot Classification**, beginning Part 2.


## Stage 1.5 applied evidence

Part 1 mechanics are complemented by optional real-data labs:

- CIFAR-10 CNN/ResNet/ViT training and backbone benchmarking
- Oxford-IIIT Pet TinyYOLO/TinyDETR/TinyUNet training
- CIFAR-10 CLIP-style, MIM, DINO and linear-probe experiments
- checkpoint-based LoRA, distillation, pruning and dynamic INT8 evaluation

These labs do not replace deterministic module tests or native parity. They provide real-data training and measured evidence while remaining optional for CPU-only learners.
