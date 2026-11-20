# Applied Lab 03 — CIFAR-10 Representation Learning

This lab adds real-data experiments for:

- **Module 26** — CLIP-style contrastive alignment
- **Module 27** — masked image modeling
- **Module 28** — DINO-style self-supervision
- **Module 29** — reusable foundation representations

The goal is not to reproduce web-scale foundation models. The goal is to make the pretraining ideas trainable, measurable, and inspectable on a public dataset.

## Dataset

CIFAR-10 is reused from Applied Lab 01.

Benefits:

- already downloaded by many learners
- 10 class names provide simple text prompts
- 50k training images are enough to show representation-learning behavior
- sample limits keep CPU experiments possible

## Experiment A — CLIP-style class-prompt alignment

The lab creates text prompts such as:

```text
a photo of an airplane
a photo of an automobile
a photo of a bird
...
```

A small text encoder and image encoder are trained into a shared embedding space.

For each image, classification is performed by similarity to all ten prompt embeddings:

```text
image
  ↓
image encoder
  ↓
normalized embedding
  ×
10 normalized prompt embeddings
  ↓
temperature-scaled similarities
  ↓
cross entropy against image class
```

This is a **small educational CLIP-style objective**, not a claim of zero-shot generalization from internet-scale language supervision.

## Experiment B — Masked image modeling

A 32×32 image is split into 4×4 patches.

Random patches are masked and a tiny transformer predicts patch pixels.

Loss is computed **only on masked patches**.

The checkpoint retains the encoder so the learned representation can be evaluated with a linear probe.

## Experiment C — DINO-style self-distillation

Each CIFAR image produces two augmented views.

```text
view A → student
view B → teacher
      ↘ cross-view loss ↙
teacher ← EMA(student)
```

The lab includes:

- student temperature
- teacher temperature
- center state
- EMA teacher update
- no labels in the self-distillation loss

Labels are used only later for linear-probe evaluation.

## Quick CPU examples

### CLIP-style

```bash
python applied_labs/03_CIFAR10RepresentationLearning/train_clip.py \
  --device cpu \
  --epochs 2 \
  --max-train-samples 5000 \
  --max-test-samples 1000 \
  --download
```

### MIM

```bash
python applied_labs/03_CIFAR10RepresentationLearning/train_mim.py \
  --device cpu \
  --epochs 2 \
  --max-train-samples 5000 \
  --download
```

### DINO-style

```bash
python applied_labs/03_CIFAR10RepresentationLearning/train_dino.py \
  --device cpu \
  --epochs 2 \
  --max-train-samples 5000 \
  --download
```

## Linear probe

After MIM or DINO pretraining:

```bash
python applied_labs/03_CIFAR10RepresentationLearning/linear_probe.py \
  --encoder dino \
  --checkpoint runs/applied_lab_03/dino/best.pt \
  --device auto \
  --epochs 10
```

The backbone is frozen. Only the linear classifier is trained.

## Evidence produced

CLIP-style:

- prompt-similarity classification accuracy
- checkpoint
- class-prompt embedding contract

MIM:

- masked reconstruction loss
- encoder checkpoint
- mask ratio / patch-size metadata

DINO:

- self-distillation loss
- EMA teacher checkpoint
- center/temperature metadata

Linear probe:

- held-out classification accuracy
- evidence that the pretrained representation carries class information

## CI policy

CI does not download CIFAR-10.

Synthetic-tensor tests verify:

- image encoder output shape
- text encoder output shape
- CLIP similarity matrix
- MIM masked-loss path
- DINO teacher/student forward path
- EMA update

## Exercises

1. Compare linear-probe accuracy for random, MIM, and DINO encoders.
2. Increase mask ratio from 0.50 to 0.75.
3. Add more text prompt templates and average their embeddings.
4. Compare teacher momentum values.
5. Visualize embeddings with PCA or t-SNE.
6. Reuse the best encoder in Applied Lab 02.
