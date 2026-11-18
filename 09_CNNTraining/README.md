# Module 09 — CNN Training Loop from Scratch

Module 08 established convolution as a numerical primitive. Module 09 turns that primitive into a **trainable vision system**.

The goal is not to hide training behind a framework helper. The goal is to make the entire lifecycle visible:

```text
dataset
  ↓
mini-batch
  ↓
forward pass
  ↓
loss
  ↓
backward pass
  ↓
optimizer step
  ↓
evaluation
  ↓
checkpoint / artifact
  ↓
native inference smoke test
```

## Learning objectives

By the end of this module you should be able to:

- build a small CNN in PyTorch
- explain the role of `train()` and `eval()`
- implement a full epoch loop
- zero gradients correctly
- compute loss and call backpropagation
- update parameters with an optimizer
- evaluate without gradient tracking
- distinguish training loss from validation accuracy
- save a model checkpoint
- export a compact inference artifact
- reproduce the same forward pass in C++17
- understand which parts of training disappear at inference time

## 1. The tiny vision task

The module uses a deterministic synthetic dataset with two visual classes:

```text
class 0: vertical bright structure
class 1: horizontal bright structure
```

Each sample is a small grayscale image.

The task is intentionally simple so the learner can inspect:

- input tensors
- convolution kernels
- logits
- loss
- gradients
- learned parameters

The objective is not benchmark performance. It is to understand the training system.

## 2. CNN architecture

The network is deliberately small:

```text
1×4×4 image
    ↓
Conv2d(1 → 2, 3×3, padding=1)
    ↓
ReLU
    ↓
Global average pooling
    ↓
Linear(2 → 2)
    ↓
class logits
```

Mathematically:

$$
F = \operatorname{ReLU}(W_c * X + b_c)
$$

Global average pooling gives:

$$
g_k =
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
F_k(i,j)
$$

Then:

$$
\mathbf{z}=W_f\mathbf{g}+\mathbf{b}_f
$$

where $\mathbf{z}$ contains the two class logits.

## 3. Why logits are kept

The training loop uses `CrossEntropyLoss`.

PyTorch cross entropy expects logits directly.

Do not insert softmax into the model before `CrossEntropyLoss`.

Conceptually:

```text
logits
   ↓
CrossEntropyLoss
   ↓
log-softmax + negative log likelihood
```

Softmax is useful for interpreting probabilities during inference, but it is not required before this loss.

## 4. One training step

The core loop is:

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

Each line has a distinct responsibility.

### `zero_grad()`

Gradients accumulate by default in PyTorch.

If gradients are not cleared, the next batch adds to the previous gradients.

### Forward pass

The current parameters produce logits.

### Loss

The loss converts prediction quality into a scalar objective.

### `backward()`

Autograd traverses the graph in reverse and populates parameter gradients.

### `step()`

The optimizer changes the parameters using those gradients.

## 5. Epoch structure

An epoch processes the training dataset once.

```text
for epoch
   for mini-batch
      zero gradients
      forward
      loss
      backward
      optimizer step

   evaluate validation set
   record metrics
```

A useful training loop reports at least:

- mean training loss
- validation loss
- validation accuracy

Later modules add richer metrics, schedulers, augmentation, and checkpoint policy.

## 6. Train mode vs evaluation mode

PyTorch modules have two operating modes:

```python
model.train()
model.eval()
```

This becomes critical when the network contains layers such as:

- dropout
- batch normalization

Even when the tiny Module 09 CNN does not depend heavily on those layers, the correct lifecycle is introduced now.

Evaluation should also disable gradient construction:

```python
with torch.no_grad():
    ...
```

This reduces memory use and avoids unnecessary autograd work.

## 7. Determinism

The module sets a fixed PyTorch seed and uses a deterministic synthetic dataset.

This does not mean all large-scale training is perfectly deterministic.

It means the reference experiment has a controlled baseline.

From Module 05, reproducibility still requires:

- dataset identity
- split definition
- random seeds
- model definition
- optimizer configuration
- software environment
- evaluation definition

## 8. Checkpoint vs inference artifact

These are different concepts.

### Training checkpoint

A checkpoint may include:

- model state
- optimizer state
- epoch
- best metric
- training configuration

It exists so training can resume.

### Inference artifact

The native C++ smoke runtime needs only what affects forward inference:

- convolution weights
- convolution bias
- classifier weights
- classifier bias
- tensor shape conventions
- artifact version

The course keeps those roles separate.

## 9. Python implementation

The module provides:

- deterministic synthetic dataset creation
- `TinyCNN`
- one-epoch training function
- evaluation function
- multi-epoch training helper
- checkpoint save/load
- native artifact export/load
- pure-Python artifact inference helper

The artifact inference helper is important because it provides a second reference path independent of the live PyTorch module object.

## 10. Native C++ inference smoke path

The C++ runtime executes:

```text
input image
   ↓
3×3 convolution
   ↓
ReLU
   ↓
global average pooling
   ↓
2-class linear layer
   ↓
logits
```

It does **not** implement training.

That separation mirrors real systems:

```text
training system             production runtime
---------------             ------------------
dataset                      sensor input
autograd                     forward only
optimizer                    no optimizer
checkpointing                inference artifact
Python/PyTorch      →        C++ runtime
```

## 11. Python ↔ C++ parity

The committed smoke artifact is executed through:

- Python artifact inference
- C++17 artifact inference

The parity gate compares both output logits.

A learner can also train a new model and export the learned parameters using the same artifact format.

## 12. Why global average pooling?

Global average pooling avoids introducing a large flattening-dependent fully connected layer.

It also teaches a pattern used in many CNN architectures:

```text
spatial features
     ↓
average each channel
     ↓
compact feature vector
     ↓
classifier
```

It makes the native runtime easier to inspect while still preserving a real CNN pipeline.

## 13. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Ruff | Required | Python training implementation |
| mypy | Required | Typed training helpers |
| pytest | Required | Training/evaluation/artifact tests |
| CPU PyTorch | Required | Training baseline |
| C++ configure/build/CTest | Required | Native inference smoke path |
| Python ↔ C++ parity | Required | Same artifact, same logits |
| ONNX | N/A | First full ONNX checkpoint remains Module 11 |
| GPU | Optional | Training API is device-aware; CPU remains CI baseline |

## 14. Module structure

```text
09_CNNTraining/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── artifacts/
│   └── tiny_cnn_smoke.txt
├── python/
│   ├── cnn_training.py
│   ├── cnn_training_walkthrough.py
│   └── cnn_training_native_parity.py
├── cpp/
│   └── tiny_cnn_inference.cpp
└── tests/
    ├── conftest.py
    └── test_cnn_training.py
```

## 15. Training loop evidence

A useful training run should show:

```text
epoch
training loss
validation loss
validation accuracy
```

Do not judge learning from one batch.

Do not judge generalization from training loss alone.

## 16. Common failure modes

### Forgetting `zero_grad()`

Gradients accumulate unintentionally.

### Calling `eval()` but not `no_grad()`

The network uses evaluation behavior, but autograd still records operations.

### Calling `no_grad()` but leaving `train()` mode enabled

BatchNorm/dropout semantics can remain in training mode.

### Applying softmax before `CrossEntropyLoss`

The loss already expects logits.

### Saving only source code

Learned parameters are part of the model.

### Copying trained weights manually into C++

That creates artifact drift and destroys traceability.

### Comparing only the predicted class

Two runtimes can produce the same class while having significantly different logits.

Parity should compare numerical outputs.

## 17. Engineering rules

1. make training phases explicit
2. clear gradients intentionally
3. use logits with cross entropy
4. evaluate under `eval()` and `no_grad()`
5. keep dataset and split deterministic for reference tests
6. distinguish checkpoints from inference artifacts
7. export learned parameters rather than hard-coding them
8. compare native logits, not only class labels
9. validate artifact dimensions
10. keep native inference free of training dependencies

## 18. Connection to the surround-perception capstone

The later surround detector will use the same lifecycle at larger scale:

```text
surround-camera dataset
       ↓
training pipeline
       ↓
CNN / Transformer backbone
       ↓
detection heads
       ↓
checkpoint
       ↓
exported runtime artifact
       ↓
C++ / GPU inference
```

Module 09 is the first complete miniature of that production path.

## 19. Book takeaway

> **Training is a stateful optimization process; inference is a forward-only numerical contract. Good ML engineering makes the boundary between them explicit.**

## 20. Exercises

1. Add mini-batches with `DataLoader`.
2. Change SGD to Adam.
3. Add a validation split.
4. Plot loss by epoch.
5. Inspect learned convolution filters.
6. Add a third synthetic pattern class.
7. Export a newly trained model and run C++ parity.
8. Save optimizer state in a checkpoint.
9. Add device selection for CUDA.
10. Explain what changes when the network is deployed at 30 FPS.

## Stage 1.5 applied lab

The deterministic module above remains the mechanics reference. For a real-data run, continue with:

- [Applied Lab 01 — CIFAR-10 Backbones](../applied_labs/01_CIFAR10Backbones/README.md)

That lab trains actual CIFAR-10 models, records held-out accuracy, writes checkpoints/plots, and keeps the dataset download optional.

## Next

Module 10 adds regularization, augmentation, learning-rate scheduling, and robust checkpoint policy.
