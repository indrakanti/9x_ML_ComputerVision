# Stage 1.5 — Applied Labs Roadmap

Stage 1.5 is the real-data companion to **ML-CV Part 1 (Modules 00–31)**.

It does not replace the deterministic mechanics modules. It adds end-to-end experiments on public research datasets.

## Status vocabulary

- **Mechanics implemented** — deterministic teaching implementation exists in Stage 1.
- **Applied lab implemented** — optional real-data training/evaluation exists.
- **Applied lab planned** — mechanics exist; real-data layer is still to be added.

## Applied Lab 01 — CIFAR-10 Backbones

**Status: Implemented**

Extends Modules:

- 09 — CNN training
- 12 — ResNet
- 14 — Vision Transformer
- 17 — backbone evaluation

Real-data evidence:

- train/validation/test workflow
- TinyCNN
- CIFAR-adapted ResNet-18
- TinyViT
- checkpoints
- learning curves
- held-out test accuracy
- parameter counts
- CPU/CUDA latency benchmark tool

## Applied Lab 02 — Detection and Segmentation

**Status: Planned**

Target mechanics:

- 18 — detection metrics
- 19 — YOLO-style one-stage detection
- 20 — two-stage detection
- 21 — DETR
- 22 — semantic segmentation
- 23 — instance/panoptic segmentation

Planned real-data layer:

- small public pedestrian/detection dataset
- end-to-end trainable YOLO-style model
- small trainable DETR
- real segmentation masks
- mAP / IoU / mIoU evidence
- visualization of boxes, masks, and matching

CPU users will have sample/epoch limits; full experiments remain optional.

## Applied Lab 03 — Foundation Representation Learning

**Status: Planned**

Target mechanics:

- 26 — CLIP-style contrastive learning
- 27 — masked image modeling
- 28 — DINO-style self-supervision
- 29 — foundation backbones

Planned real-data layer:

- CIFAR-10 / STL-10-style public image data
- zero-shot prompt experiment
- masked pretraining experiment
- teacher/student self-supervised experiment
- linear-probe evaluation
- embedding visualization

## Applied Lab 04 — Adaptation and Compression

**Status: Planned**

Target mechanics:

- 30 — PEFT / LoRA
- 31 — distillation, pruning, quantization

Input models:

- checkpoints produced by Applied Lab 01 and/or Lab 03

Planned evidence:

- frozen linear probe
- LoRA adaptation
- parameter-count comparison
- pruning sparsity
- FP32 vs quantized accuracy
- latency/model-size comparison on the actual runtime

## Optional visualization notebooks

Notebooks are companions, not canonical implementations.

Planned notebooks:

1. CIFAR samples and augmentations
2. CNN feature maps and filters
3. ViT patch/token/attention visualization
4. detection boxes + NMS
5. DETR cost/matching visualization
6. semantic/instance masks
7. CLIP similarity matrix and embeddings
8. DINO/MIM representation visualization
9. quantization error distributions

Canonical reusable code stays in Python modules/scripts so it remains lintable, testable, and importable.

## Completion rule

Stage 1.5 is complete when Labs 01–04 provide real-data evidence for the highest-value Part-1 model families while the root CPU CI remains network-independent.
