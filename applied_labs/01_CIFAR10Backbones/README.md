# Applied Lab 01 — CIFAR-10 Backbones

This lab adds the real-data layer for:

- **Module 09** — CNN training loop
- **Module 12** — ResNet
- **Module 14** — Vision Transformer
- **Module 17** — backbone evaluation

The mechanics modules remain the source of first-principles explanations and native parity. This lab answers a different question:

> What happens when we train these ideas end to end on a real public image dataset?

## Dataset

CIFAR-10 contains 32×32 RGB images in 10 classes and is available directly through torchvision.

The lab uses:

- official CIFAR-10 training split
- deterministic train/validation split inside the training set
- official CIFAR-10 test split
- random crop + horizontal flip for training
- tensor conversion + CIFAR-style normalization
- no network access in CI

## Models

### TinyCNN

A deliberately small CNN:

```text
3×32×32
→ Conv/ReLU
→ Conv/ReLU/Pool
→ Conv/ReLU
→ Global Average Pool
→ Linear(10)
```

### CIFAR-ResNet18

Torchvision ResNet-18 adapted for 32×32 images:

- 3×3 stride-1 stem
- no initial max-pool
- 10-class final layer
- no pretrained-weight download

### TinyViT

A compact trainable ViT:

```text
3×32×32
→ 4×4 patch embedding
→ class token + learned positions
→ TransformerEncoder
→ LayerNorm
→ Linear(10)
```

This is intentionally smaller than production ViTs so CPU learners can run it.

## Quick CPU experiment

Use a subset first:

```bash
python applied_labs/01_CIFAR10Backbones/train.py \
  --model tinycnn \
  --device cpu \
  --epochs 2 \
  --max-train-samples 5000 \
  --max-val-samples 1000 \
  --max-test-samples 1000 \
  --download
```

## Full TinyCNN run

```bash
python applied_labs/01_CIFAR10Backbones/train.py \
  --model tinycnn \
  --epochs 20 \
  --device auto \
  --download
```

## ResNet-18

```bash
python applied_labs/01_CIFAR10Backbones/train.py \
  --model resnet18 \
  --epochs 40 \
  --device auto \
  --download
```

## TinyViT

```bash
python applied_labs/01_CIFAR10Backbones/train.py \
  --model tinyvit \
  --epochs 40 \
  --device auto \
  --download
```

These epoch counts are starting points, not benchmark claims.

## Output directory

By default:

```text
runs/applied_lab_01/<model>/
├── best.pt
├── last.pt
├── metrics.json
└── learning_curve.png
```

The checkpoint stores:

- model name
- model state
- optimizer state
- epoch
- seed
- best validation accuracy
- dataset/split metadata

## Benchmark a trained checkpoint

```bash
python applied_labs/01_CIFAR10Backbones/benchmark.py \
  --checkpoint runs/applied_lab_01/resnet18/best.pt \
  --device auto \
  --batch-size 1 \
  --warmup 20 \
  --iterations 100
```

For CUDA, the benchmark synchronizes around timed regions. Reported latency is still hardware/software-specific.

## Compare experiments

Train multiple models, then run:

```bash
python applied_labs/01_CIFAR10Backbones/compare.py \
  runs/applied_lab_01/tinycnn/metrics.json \
  runs/applied_lab_01/resnet18/metrics.json \
  runs/applied_lab_01/tinyvit/metrics.json
```

This prints a compact table using real held-out test accuracy and parameter counts.

Latency belongs in the benchmark report because it depends on the target device.

## What to record

For a meaningful comparison, keep fixed:

- dataset split
- augmentation
- optimizer policy where practical
- epoch budget
- batch size or effective batch size
- image resolution
- precision
- device
- software versions

Architecture comparison is misleading when every row uses a different training recipe without saying so.

## Exercises

1. Increase TinyViT depth from 2 to 4 and compare accuracy/latency.
2. Add MixUp or CutMix and record the training-policy change.
3. Compare ResNet-18 with MobileNetV3.
4. Add AMP on CUDA and compare throughput.
5. Export the best model to ONNX and connect it to the Module-11 export workflow.
6. Plot a confusion matrix for the test set.
