# Module 11 — Transfer Learning + First ONNX Export Checkpoint

Module 11 closes the first CNN-foundations arc by connecting training to a portable model format.

The module demonstrates three ideas together:

1. reuse a pretrained feature extractor
2. train only a new task-specific head
3. export the resulting model and prove the same inference numerically in multiple runtimes

The central contract is:

```text
pretrained backbone
      ↓
freeze backbone
      ↓
train new classifier head
      ↓
PyTorch reference
      ↓
ONNX export
      ↓
ONNX Runtime
      ↓
native C++17 reference
      ↓
three-way logit parity
```

## Learning objectives

You should be able to:

- explain why transfer learning can reduce data and training requirements
- distinguish feature extraction from fine-tuning
- freeze and unfreeze model parameters intentionally
- train only a classification head
- verify that frozen parameters did not change
- export a PyTorch model to ONNX
- validate an ONNX graph
- run ONNX Runtime inference
- compare PyTorch and ONNX Runtime logits
- execute the same forward contract in C++17
- understand what ONNX solves and what it does not solve

## 1. Why transfer learning?

Training every model from random initialization is expensive.

A pretrained backbone has already learned useful visual features:

```text
edges
textures
shapes
parts
semantic structure
```

Transfer learning reuses those features for a new task.

For a classifier:

```text
pretrained backbone
        ↓
feature vector
        ↓
new classifier head
```

Only the head may need to be trained initially.

## 2. Feature extraction vs fine-tuning

### Feature extraction

Freeze the backbone:

```python
for parameter in backbone.parameters():
    parameter.requires_grad = False
```

Then optimize only the new head.

### Fine-tuning

Later, some or all backbone parameters may be unfrozen and trained with a smaller learning rate.

This module starts with feature extraction because it makes the parameter boundary explicit.

## 3. Offline reproducible pretrained backbone

A production course should not require CI to download external weights.

This module therefore ships a tiny deterministic backbone artifact.

It represents the same engineering idea as a downloaded pretrained model:

```text
pretrained parameters
       ↓
load
       ↓
freeze
       ↓
reuse
```

The artifact contains two 3×3 filters and biases.

The filters are intentionally interpretable:

- one responds more strongly to vertical structure
- one responds more strongly to horizontal structure

The transfer task trains only the 2-class linear head.

## 4. Tiny transfer architecture

```text
1×4×4 image
    ↓
frozen Conv2d(1 → 2, 3×3, padding=1)
    ↓
ReLU
    ↓
global average pooling
    ↓
trainable Linear(2 → 2)
    ↓
logits
```

The backbone output is:

$$
F=\operatorname{ReLU}(W_b * X+b_b)
$$

Global average pooling gives:

$$
g_k=
\frac{1}{HW}
\sum_{i,j}F_k(i,j)
$$

The trainable head computes:

$$
\mathbf{z}=W_h\mathbf{g}+\mathbf{b}_h
$$

## 5. Freezing is an optimization contract

A frozen parameter has:

```text
requires_grad = False
```

and should not be passed to the optimizer.

The module verifies:

- backbone tensors before training
- backbone tensors after training

They must be identical.

This catches accidental fine-tuning.

## 6. Head-only training

The optimizer receives only:

```python
model.classifier.parameters()
```

The training loop remains the same:

```text
forward
loss
backward
optimizer step
```

but the optimizer updates only the task-specific head.

## 7. Why ONNX?

ONNX stores a computational graph and tensor parameters in a framework-neutral format.

The intended transition is:

```text
PyTorch training code
       ↓
ONNX graph
       ↓
runtime independent of PyTorch training process
```

ONNX does **not** automatically solve:

- preprocessing mismatch
- unsupported operators
- numerical drift
- dynamic-shape mistakes
- latency
- memory ownership
- scheduling
- safety monitoring

Export is the beginning of deployment validation, not the end.

## 8. Export

The module exports with an explicit input/output contract:

```text
input  : [N, 1, 4, 4]
output : [N, 2]
```

The graph uses a fixed spatial shape but a dynamic batch dimension.

The export path validates the graph with `onnx.checker`.

## 9. ONNX Runtime inference

The exported graph is loaded with CPU ONNX Runtime.

The exact same input is executed through:

- PyTorch
- ONNX Runtime

The logits must match within a strict numerical tolerance.

## 10. Native C++17 reference

This module also retains the course's native-runtime discipline.

The C++ reference loads the same language-neutral parameter artifact and executes:

```text
3×3 convolution
ReLU
global average pooling
linear head
```

This is intentionally not yet ONNX Runtime C++ linkage.

That later module will teach:

- session creation
- allocator ownership
- tensor construction
- execution providers
- runtime libraries
- packaging

Module 11 instead proves the model itself is portable.

## 11. Three-way parity

The committed transfer-model artifact is loaded into the PyTorch model.

Then:

```text
                 ┌→ PyTorch
same model/input ├→ ONNX Runtime
                 └→ C++17
```

All three must produce the same two logits.

This gives two independent checks:

1. framework export parity
2. native mathematical parity

## 12. Model artifact vs ONNX artifact

The course uses both here for a reason.

### Parameter artifact

Human-readable and easy to inspect.

Useful for:

- teaching
- native reference implementation
- debugging weights/layout

### ONNX

Machine-readable executable graph.

Useful for:

- portable runtime execution
- graph inspection
- deployment toolchains

For real models, ONNX becomes the primary deployable graph.

## 13. Runtime applicability

| Gate | Status |
|---|---|
| Ruff | Required |
| mypy | Required |
| pytest | Required |
| PyTorch transfer-learning test | Required |
| frozen-backbone integrity test | Required |
| ONNX checker | Required |
| PyTorch ↔ ONNX Runtime parity | Required |
| C++ configure/build/CTest | Required |
| PyTorch ↔ C++ parity | Required |
| ONNX Runtime C++ linkage | Deferred to production-runtime module |
| GPU | Optional |

## 14. Module structure

```text
11_TransferLearningONNX/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── artifacts/
│   ├── pretrained_backbone.txt
│   └── transfer_model_smoke.txt
├── python/
│   ├── transfer_learning.py
│   ├── transfer_learning_walkthrough.py
│   └── transfer_learning_native_parity.py
├── cpp/
│   └── transfer_model.cpp
└── tests/
    ├── conftest.py
    └── test_transfer_learning.py
```

## 15. Failure modes

- accidentally optimizing frozen parameters
- forgetting to switch to evaluation mode before export
- exporting preprocessing assumptions only in documentation
- comparing only final class labels
- using a different tensor layout in native code
- silently changing class order
- assuming successful ONNX export guarantees runtime support
- forgetting ONNX graph validation
- ignoring numerical parity
- downloading mutable external weights in CI

## 16. Engineering rules

1. version pretrained weights
2. freeze intentionally
3. verify frozen tensors remain unchanged
4. train only intended parameters
5. export in evaluation mode
6. define input/output names and shapes
7. validate the ONNX graph
8. compare logits numerically
9. preserve class ordering and preprocessing metadata
10. keep an independent reference path for debugging

## 17. Connection to the surround-camera capstone

A later detector will commonly begin from a pretrained backbone.

The same pattern scales directly:

```text
pretrained visual backbone
        ↓
task-specific detection heads
        ↓
fine-tuning
        ↓
validated checkpoint
        ↓
ONNX / optimized runtime
        ↓
C++ streaming deployment
```

Module 11 is the first complete miniature of that workflow.

## 18. Book takeaway

> **Transfer learning reuses representation knowledge; ONNX transfers the forward graph across runtimes. Neither is trustworthy without an explicit numerical equivalence test.**

## 19. Exercises

1. Unfreeze the last convolution filter and fine-tune it.
2. Use different learning rates for backbone and head.
3. Add class-name metadata.
4. Export with a dynamic batch size.
5. Inspect ONNX graph nodes.
6. Intentionally transpose the classifier weight and observe parity failure.
7. Change preprocessing and measure the effect.
8. Add an ONNX model checksum to a manifest.
9. Compare float32 and float64 parity tolerance.
10. Replace the tiny backbone with a torchvision model outside CI.

## Next

Module 12 introduces residual learning and ResNet.
