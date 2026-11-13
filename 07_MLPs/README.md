# Module 07 — MLPs, Activations, Initialization & Normalization

A linear classifier can only draw one hyperplane. An MLP becomes useful because it composes linear layers with nonlinear activations.

This chapter builds a small multilayer perceptron from first principles, maps it to PyTorch, and deploys the same forward pass in C++17 from a portable artifact.

## Learning objectives

You should be able to:

- explain why stacked linear layers without activations collapse into one linear transform
- implement ReLU and understand its gradient
- explain hidden layers and nonlinear decision boundaries
- understand Xavier/Glorot and He/Kaiming initialization intuitively
- explain why normalization stabilizes optimization
- implement a 2-layer MLP forward pass
- compare hand-written math with PyTorch
- serialize MLP parameters and normalization metadata
- reproduce inference in C++17
- prove Python ↔ C++ parity

## 1. Why nonlinearity matters

Two linear transforms without an activation are still linear:

$$
\mathbf{y}=W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2
$$

which can be rearranged into:

$$
\mathbf{y}=W'\mathbf{x}+\mathbf{b}'
$$

So depth alone does not create nonlinear modeling power.

An MLP inserts an activation:

$$
\mathbf{h}=\phi(W_1\mathbf{x}+\mathbf{b}_1)
$$

$$
z=W_2\mathbf{h}+b_2
$$

## 2. ReLU

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

Its derivative is:

$$
\frac{d}{dx}\operatorname{ReLU}(x)=
\begin{cases}
1 & x>0 \\
0 & x<0
\end{cases}
$$

ReLU is simple, cheap, and historically important in deep vision models.

## 3. Initialization

Poor initialization can make activations or gradients shrink or explode.

### Xavier / Glorot

A common scale is based on both fan-in and fan-out:

$$
\operatorname{Var}(W)\approx\frac{2}{n_{in}+n_{out}}
$$

### He / Kaiming

For ReLU networks, a common choice is:

$$
\operatorname{Var}(W)\approx\frac{2}{n_{in}}
$$

The goal is not a magical constant. The goal is to preserve useful signal scale across layers.

## 4. Normalization

This module uses fixed feature normalization:

$$
x_i' = \frac{x_i-\mu_i}{\sigma_i}
$$

The same normalization must be used in training and deployment.

That makes preprocessing part of the model contract.

## 5. Forward pass

For normalized input $\mathbf{x}'$:

$$
\mathbf{a}_1=W_1\mathbf{x}'+\mathbf{b}_1
$$

$$
\mathbf{h}=\operatorname{ReLU}(\mathbf{a}_1)
$$

$$
z=W_2\mathbf{h}+b_2
$$

$$
p=\sigma(z)
$$

## 6. Portable artifact

The module artifact contains:

```text
version
input_size
hidden_size
normalization mean/std
W1 / b1
W2 / b2
threshold
```

This extends the deployment contract introduced in Module 06.

## 7. Runtime applicability

| Gate | Status |
|---|---|
| Ruff | Required |
| mypy | Required |
| pytest | Required |
| CPU PyTorch equivalence | Required |
| C++ configure/build/CTest | Required |
| Python ↔ C++ parity | Required |
| ONNX | N/A — first full neural export checkpoint remains Module 11 |
| GPU | N/A — architecture is intentionally tiny |

## 8. Structure

```text
07_MLPs/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── artifacts/mlp_binary_model.txt
├── python/
│   ├── mlp.py
│   ├── mlp_walkthrough.py
│   └── mlp_native_parity.py
├── cpp/mlp.cpp
└── tests/
    ├── conftest.py
    └── test_mlp.py
```

## 9. Python reference

The Python implementation covers:

- normalization
- affine layers
- ReLU
- sigmoid
- two-layer forward pass
- He initialization helper
- artifact save/load

## 10. PyTorch equivalence

The same artifact parameters are loaded into:

```python
torch.nn.Sequential(
    torch.nn.Linear(4, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1),
)
```

The hand-written and PyTorch logits must agree.

## 11. C++17 counterpart

The native implementation:

- parses the artifact
- validates shapes
- normalizes input
- executes dense layer 1
- applies ReLU
- executes dense layer 2
- applies sigmoid and threshold
- emits JSON for parity testing

## 12. Worked example

The supplied artifact uses 4 inputs and 3 hidden units.

The important engineering chain is:

```text
same artifact
   ├── Python forward
   ├── PyTorch forward
   └── C++17 forward
           ↓
        same logit
```

## 13. Failure modes

- omitting activation: network collapses to a linear model
- train/deploy normalization mismatch
- bad initialization causing unstable scale
- wrong matrix orientation between Python and C++
- silently changing threshold
- artifact shape mismatch
- comparing probabilities when logits already disagree

## 14. Engineering rules

1. preprocessing belongs in the deployable contract
2. matrix layout and shape must be explicit
3. activation semantics must match across runtimes
4. initialization should be intentional
5. compare logits before probabilities
6. version artifacts
7. validate dimensions in native code
8. keep deterministic CPU reference paths
9. add ONNX only when it adds real model-runtime value
10. maintain Python ↔ C++ parity as models grow

## 15. Connection to the autonomous-driving capstones

MLPs appear inside larger perception and prediction systems:

- detector classification/regression heads
- fusion projection layers
- tracking association networks
- trajectory-prediction decoders

The architecture will become larger, but the same concerns remain: shapes, normalization, activations, parameters, and runtime equivalence.

## 16. Book takeaway

> **Nonlinearity gives neural networks expressive power; initialization and normalization help that power remain trainable; deployment requires those semantics to survive outside Python.**

## 17. Exercises

1. Replace ReLU with tanh.
2. Add a second hidden layer.
3. Compare Xavier vs He initialization statistics.
4. Intentionally transpose a weight matrix and observe parity failure.
5. Add feature names to the artifact.
6. Add hidden activation statistics.
7. Add float32 inference and compare with float64.
8. Extend the artifact to multiclass softmax.
9. Add checksum/version validation.
10. Explain how this MLP maps to a detector head.

## Next

Module 08 introduces convolution from first principles and begins the first meaningful CPU/GPU execution comparison.
