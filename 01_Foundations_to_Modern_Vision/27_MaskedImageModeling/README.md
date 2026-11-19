# Module 27 — Masked Image Modeling

Masked image modeling learns visual representations by hiding part of an image and asking the model to reconstruct the missing content.

A compact MAE-style view is:

```text
image
  ↓
patchify
  ↓
mask selected patches
  ↓
encode visible patches
  ↓
decode / reconstruct masked patches
  ↓
loss on masked targets only
```

## Learning objectives

You should be able to:

- explain masked-image pretraining
- convert an image into patch tokens
- distinguish visible and masked patches
- explain why reconstruction loss is usually applied only to masked content
- compute masked mean-squared error
- understand encoder/decoder asymmetry in MAE-style models
- reproduce patch selection and masked reconstruction in C++17
- reason about mask ratio, token count, and pretraining compute

## 1. Why mask the image?

If the model sees every pixel, trivial identity mappings become possible.

Masking forces the representation to use surrounding context.

For patch set:

$$
X = \{x_1,x_2,\dots,x_N\}
$$

split into visible and masked subsets:

$$
X = X_{vis} \cup X_{mask}
$$

The model encodes only or primarily the visible content and reconstructs the masked target.

## 2. Patchification

For image size $H\times W$ and square patch size $P$:

$$
N=\frac{H}{P}\frac{W}{P}
$$

This module uses a 4×4 grayscale image with 2×2 patches, producing four patches.

Patch traversal order is explicit and identical to the ViT module.

## 3. Masking contract

The artifact provides deterministic masked patch indices.

For example:

```text
patches: 0 1 2 3
mask:      X   X
visible: 0   2
masked:    1   3
```

A real pretraining pipeline samples masks randomly, but deterministic masks make CI and numerical parity reproducible.

## 4. Reconstruction

The teaching reference uses a small linear predictor from the mean visible patch representation.

That is intentionally much simpler than a real masked autoencoder decoder, but it preserves the systems contract:

- visible-patch representation
- masked-patch targets
- reconstruction tensor
- masked-only loss

## 5. Masked MSE

If $M$ is the set of masked patches:

$$
L=
\frac{1}{|M|P^2C}
\sum_{i\in M}
\|\hat{x}_i-x_i\|_2^2
$$

Visible patches do not contribute to the reconstruction loss.

## 6. MAE encoder/decoder asymmetry

Masked autoencoders often use:

- large encoder for representation learning
- lightweight decoder used only during pretraining

At downstream inference time, the decoder can disappear.

This is a crucial deployment distinction: **pretraining graph != deployment graph**.

## 7. Executable reference

The artifact contains:

- image dimensions
- patch size
- mask indices
- linear reconstruction weights and bias

Python computes:

- patch tensor
- visible-patch mean
- reconstruction for each masked patch
- masked MSE

PyTorch verifies the same reconstruction/loss.

C++17 reproduces the exact output.

## 8. Deployment implications

Masked-image modeling is typically a **pretraining strategy**, not a deployed runtime path.

The deployment artifact normally retains:

```text
pretrained encoder
      ↓
fine-tuned / adapted downstream model
```

The reconstruction decoder and masking machinery can be training-only.

## 9. Failure modes

- inconsistent patch order
- training on visible patches accidentally
- applying loss to the wrong mask
- mask indices out of range
- changing pixel normalization between pretraining and fine-tuning
- assuming the pretraining decoder belongs in production inference
- comparing models pretrained with different mask ratios without labeling them

## 10. Engineering rules

1. version patch size and traversal order
2. version mask ratio/sampling policy
3. keep masked-only loss explicit
4. separate pretraining-only components from deployment components
5. preserve input normalization between pretraining and transfer
6. compare reconstructed masked patches numerically

## 11. AV relevance

Self-supervised masked pretraining can exploit huge unlabeled driving-video datasets. The downstream value comes from the learned encoder, not from deploying a reconstruction decoder in the vehicle.

## 12. Book takeaway

> **Masked image modeling turns missing visual content into a training signal. The important systems boundary is that the expensive reconstruction task can be pretraining-only while the learned encoder is reused downstream.**

## Reproducible commands

```bash
python 27_MaskedImageModeling/python/mim_walkthrough.py
pytest 27_MaskedImageModeling/tests
cmake -S 27_MaskedImageModeling -B build-m27 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m27
ctest --test-dir build-m27 --output-on-failure
python 27_MaskedImageModeling/python/mim_native_parity.py \
  build-m27/cv9x_mim_reconstruct \
  27_MaskedImageModeling/artifacts/mim.txt
```

## Stage 1.5 applied lab

The deterministic module above remains the mechanics reference. The real-data companion is:

- [Applied Lab 03 — CIFAR-10 Representation Learning](../applied_labs/03_CIFAR10RepresentationLearning/README.md)

It trains compact CLIP-style, masked-image, and DINO-style representations on real CIFAR-10 images and evaluates frozen features with a linear probe.

## Next

Module 28 introduces DINO-style self-distillation and teacher–student self-supervision.
