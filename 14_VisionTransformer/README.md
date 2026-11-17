# Module 14 — Vision Transformer Fundamentals

Vision Transformers reinterpret an image as a sequence of patch tokens.

The core pipeline is:

```text
image
  ↓
patchify
  ↓
linear patch projection
  ↓
positional information
  ↓
self-attention
  ↓
token representations
```

This module implements the smallest useful ViT-style forward path in Python/PyTorch and C++17.

## Learning objectives

You should be able to:

- convert an image into fixed-size patches
- compute token count from image and patch size
- project flattened patches into an embedding space
- explain why positional information is required
- derive scaled dot-product attention
- compute queries, keys, values, scores, softmax, and weighted values
- reproduce token outputs in C++17
- explain quadratic attention cost
- compare CNN and ViT inductive biases
- reason about ViT deployment memory and latency

## 1. Patchification

For image height $H$, width $W$, and square patch size $P$:

$$
N=\frac{H}{P}\frac{W}{P}
$$

patches are produced.

A 4×4 image with 2×2 patches creates four tokens. Each grayscale patch contains four values.

## 2. Patch projection

Flatten a patch:

$$
\mathbf{p}\in\mathbb{R}^{P^2C}
$$

and project it:

$$
\mathbf{x}=W_e\mathbf{p}+\mathbf{b}_e
$$

The result is a token embedding.

## 3. Positional information

Self-attention alone does not know where a token came from.

Therefore positional information is added:

$$
\mathbf{x}_i'=\mathbf{x}_i+\mathbf{e}^{pos}_i
$$

## 4. Query, key, and value

For token matrix $X$:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

## 5. Scaled dot-product attention

Attention scores are:

$$
A=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)
$$

The output is:

$$
Z=AV
$$

Scaling prevents dot products from growing excessively with embedding dimension.

## 6. Tiny executable model

```text
4×4 grayscale image
      ↓
four 2×2 patches
      ↓
embedding dimension = 2
      ↓
fixed positional embeddings
      ↓
single-head self-attention
      ↓
4 tokens × 2 values
```

The executable reference intentionally omits the MLP block, LayerNorm, class token, and multiple heads so the attention arithmetic remains visible.

## 7. Class token and multi-head extension

Many ViT classifiers prepend a learned `[CLS]` token and classify from its final representation.

Real ViTs also use multiple attention heads:

```text
same token sequence
   ├→ head 1
   ├→ head 2
   └→ head h
        ↓
concatenate
        ↓
output projection
```

## 8. Quadratic token cost

Attention-score storage is approximately:

$$
O(N^2)
$$

If image resolution doubles in both dimensions while patch size stays fixed, token count grows by four and attention-score count grows by sixteen.

This is a major deployment concern.

## 9. CNN vs ViT inductive bias

CNN:

- local receptive fields
- weight sharing
- strong spatial inductive bias

ViT:

- global token interaction
- weaker built-in locality
- often benefits strongly from scale and pretraining

Architecture choice depends on data, compute, runtime, and task.

## 10. Python/PyTorch reference

The module implements:

- patchify
- patch projection
- positional addition
- Q/K/V projection
- scaled scores
- row-wise softmax
- weighted value aggregation

The PyTorch tensor implementation must match the manual reference.

## 11. C++17 counterpart

The native implementation loads the same artifact and executes the exact token pipeline.

Parity compares every output token element.

## 12. Deployment implications

ViT inference introduces memory patterns that differ from convolution:

- token matrices
- Q/K/V buffers
- attention-score matrices
- softmax
- matrix multiplications
- sequence-length sensitivity

For high-resolution perception, token count can dominate both memory and latency.

## 13. Failure modes

- patch-order mismatch
- HWC/CHW confusion
- missing positional information
- incorrect $1/\sqrt{d}$ scaling
- softmax on the wrong axis
- transposing Q/K incorrectly
- comparing only pooled outputs
- underestimating $N^2$ memory

## 14. Engineering rules

1. specify patch traversal order
2. specify flattened patch layout
3. specify embedding dimension
4. preserve positional-embedding order
5. scale attention correctly
6. apply softmax row-wise
7. compare complete token tensors
8. track token count as a deployment variable

## 15. Autonomous-driving relevance

Transformer backbones and attention-based detectors increasingly appear in perception stacks.

A high-resolution multi-camera system can create:

```text
many cameras × many patches × attention
```

which can become a substantial compute and memory load.

## 16. Book takeaway

> **A Vision Transformer converts spatial image structure into a token sequence, then learns relationships through attention; global interaction comes with explicit token-count and memory costs.**

## 17. Exercises

1. Add a class token.
2. Add LayerNorm.
3. Add a second attention head.
4. Add an MLP block.
5. Increase image size and compute attention-matrix growth.
6. Compare token count for patch sizes 8, 16, and 32.
7. Implement an output projection.
8. Add a residual connection around attention.

## Next

Module 15 introduces hierarchical and windowed transformers.
