# Module 30 — Linear Probing, Adapters, LoRA and PEFT

Large pretrained backbones are often adapted without updating every parameter.

Parameter-efficient fine-tuning (PEFT) asks:

> How much task-specific capacity can we add while keeping the expensive pretrained model mostly or completely frozen?

This module makes three common adaptation patterns executable:

```text
frozen features
   ├→ linear probe
   ├→ residual bottleneck adapter
   └→ LoRA low-rank weight update
```

## Learning objectives

You should be able to:

- distinguish linear probing from fine-tuning
- explain residual adapters
- derive a LoRA low-rank update
- compute trainable parameter counts
- explain the LoRA scaling factor
- compare PEFT overhead with full fine-tuning
- reproduce probe/adapter/LoRA outputs in C++17
- reason about which weights belong in deployment artifacts

## 1. Linear probing

A linear probe freezes the backbone and trains only a task head.

Given frozen feature vector $f$:

$$
y = W_pf + b_p
$$

This tests how linearly separable the representation already is.

A good probe result does not prove the representation is optimal for full fine-tuning, but it is a useful diagnostic.

## 2. Residual adapters

A bottleneck adapter inserts a small trainable branch:

$$
h = W_{down}x+b_{down}
$$

$$
a = \operatorname{ReLU}(h)
$$

$$
\Delta x = W_{up}a+b_{up}
$$

$$
y=x+\Delta x
$$

The pretrained path is preserved through the residual connection.

## 3. LoRA

LoRA freezes a base weight matrix $W$ and learns a low-rank update:

$$
W' = W + \frac{\alpha}{r}BA
$$

where:

- $A\in\mathbb{R}^{r\times d_{in}}$
- $B\in\mathbb{R}^{d_{out}\times r}$
- $r$ is the rank
- $\alpha$ controls update scale

Inference can either:

1. keep the LoRA branch separate, or
2. merge $BA$ into the base weight offline.

## 4. Parameter efficiency

A full linear layer contains:

$$
d_{out}d_{in}+d_{out}
$$

parameters.

LoRA adds:

$$
r(d_{in}+d_{out})
$$

trainable parameters if the base bias remains frozen.

When $r\ll d$, the trainable fraction can be much smaller.

## 5. Executable reference

The artifact contains:

- one frozen feature vector
- linear-probe weights/bias
- bottleneck-adapter down/up weights/bias
- one frozen base linear layer
- LoRA A/B matrices
- rank and alpha

Python computes:

- probe logits
- adapter output
- LoRA output
- trainable parameter counts
- trainable/full parameter ratio

PyTorch independently verifies the LoRA forward path.

C++17 reproduces all outputs and counts.

## 6. What gets deployed?

Linear probe:
- backbone
- probe head

Adapter:
- backbone
- adapter weights
- downstream head if present

LoRA:
- base model + LoRA matrices, or
- merged base weights

The deployment manifest should record whether LoRA was merged.

## 7. Failure modes

- accidentally optimizing frozen weights
- transposing LoRA A/B
- forgetting $\alpha/r$ scaling
- merging LoRA twice
- loading an adapter against the wrong backbone revision
- reporting only trainable parameters while ignoring total runtime memory
- assuming a linear probe is equivalent to full task fine-tuning

## 8. Engineering rules

1. version the base backbone
2. version feature dimension and adapter insertion point
3. log which parameters are trainable
4. record LoRA rank and alpha
5. verify frozen weights remain unchanged
6. define merged-vs-unmerged deployment state
7. compare final numerical outputs after any merge

## 9. AV relevance

PEFT is useful when a large shared visual backbone must adapt to new domains or tasks without duplicating or retraining the full model. For safety-critical systems, adapter isolation, provenance, and compatibility with the frozen base must be explicit.

## 10. Book takeaway

> **PEFT changes the optimization surface without necessarily changing the backbone. That makes compatibility metadata—base revision, insertion point, rank, scaling, and merge state—part of the model contract.**

## Reproducible commands

```bash
python 30_PEFT/python/peft_walkthrough.py
pytest 30_PEFT/tests
cmake -S 30_PEFT -B build-m30 -G Ninja -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m30
ctest --test-dir build-m30 --output-on-failure
python 30_PEFT/python/peft_native_parity.py \
  build-m30/cv9x_peft_math \
  30_PEFT/artifacts/peft.txt
```

## Stage 1.5 applied lab

The deterministic module above remains the mechanics reference. The real-model companion is:

- [Applied Lab 04 — PEFT, Distillation, Pruning and Quantization](../applied_labs/04_AdaptationCompression/README.md)

It consumes trained CIFAR-10 checkpoints and measures adaptation/compression effects on accuracy, trainable parameters, sparsity, serialized size, and CPU latency.

## Next

Module 31 compresses trained models through distillation, pruning, and quantization.
