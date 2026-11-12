# Module 04 — Autograd from First Principles to PyTorch

This module explains automatic differentiation by building a tiny scalar computation graph and then validating it against PyTorch autograd.

## Learning objectives

By the end of this module you should be able to:

- explain the difference between forward values and backward gradients
- apply the chain rule through a computation graph
- understand local derivatives and gradient accumulation
- explain why shared graph branches require gradient accumulation
- build a tiny reverse-mode autodiff engine
- compare analytical, numerical and PyTorch gradients
- understand `requires_grad`, `.grad`, `backward()` and `zero_grad()`
- recognize detached graphs and in-place/autograd hazards

## 1. Why autograd matters

Training repeatedly computes a scalar loss and asks:

> How should every parameter change to reduce that loss?

For a model parameter vector $\theta$ and loss $L$:

$$
\nabla_\theta L
=
\left[
\frac{\partial L}{\partial \theta_1},
\frac{\partial L}{\partial \theta_2},
\ldots
\right]
$$

Manual differentiation does not scale to large neural networks. Reverse-mode autodiff computes these derivatives efficiently by reusing the graph created during the forward pass.

## 2. Chain rule

Suppose

$$
a = x y
$$

and

$$
L = a^2
$$

Then:

$$
\frac{\partial L}{\partial a}=2a
$$

$$
\frac{\partial a}{\partial x}=y
$$

so:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial x}
=
2ay
$$

Autograd automates exactly this repeated local chain-rule application.

## 3. Reverse-mode intuition

Forward pass:

~~~text
x ----       multiply ---> a ---> square ---> L
y ----/
~~~

Backward pass:

~~~text
dL/dL = 1
   ↓
dL/da
   ↓
dL/dx, dL/dy
~~~

The key idea is:

> Each node knows how to propagate an incoming gradient to its parents.

## 4. Gradient accumulation

If a value contributes to the loss through multiple paths, the gradients add.

For:

$$
L=x^2+x
$$

the derivative is:

$$
\frac{dL}{dx}=2x+1
$$

The two graph paths contribute independently and must be accumulated.

This is why autograd systems use operations conceptually equivalent to:

~~~python
parent.grad += local_contribution
~~~

rather than assignment.

## 5. Numerical gradient checking

A central-difference approximation is:

$$
\frac{df}{dx}
\approx
\frac{f(x+h)-f(x-h)}{2h}
$$

Numerical gradients are slower and approximate, but they are extremely useful for testing analytic/autograd implementations.

## 6. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Python formatting / lint | Required | Autodiff engine and walkthrough |
| mypy | Required | Typed scalar graph implementation |
| pytest | Required | Graph, accumulation and gradient checks |
| CPU PyTorch tests | Required | First-principles gradients are checked against PyTorch |
| C++ configure/build/CTest | Required | Reverse-mode autodiff is implemented as a native graph |
| Python ↔ C++ numerical parity | Required | Forward value and gradients must match |
| ONNX tests | N/A | No exported model artifact |
| Python ↔ exported-model equivalence | N/A | No exported model |

## 7. What this module implements

~~~text
04_Autograd/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── python/
│   ├── autodiff.py
│   ├── autograd_walkthrough.py
│   └── native_parity.py
├── cpp/
│   └── autodiff.cpp
└── tests/
    ├── conftest.py
    └── test_autodiff.py
~~~

The scalar engine supports:

- addition
- multiplication
- power
- negation/subtraction
- reverse topological traversal
- accumulated gradients

## 8. Run

~~~bash
source .venv/bin/activate

python 04_Autograd/python/autograd_walkthrough.py
pytest 04_Autograd/tests -q

cmake -S 04_Autograd -B build/module04 -DCMAKE_BUILD_TYPE=Release
cmake --build build/module04 --parallel
ctest --test-dir build/module04 --output-on-failure

python 04_Autograd/python/native_parity.py \
  build/module04/cv9x_autodiff
~~~

## 9. Native C++ counterpart

The C++17 implementation builds the same reverse-mode graph using native nodes, parent edges, local backward functions, topological traversal and gradient accumulation.

The proof chain for this module is:

```text
first-principles Python autodiff
          ↕ parity
first-principles C++17 autodiff
          ↕ gradient equivalence
PyTorch autograd
```

This is not yet model deployment, but it demonstrates that the mathematical engine can be reproduced and verified outside Python.

## 10. PyTorch mapping

The tiny engine maps directly to PyTorch concepts:

| Tiny engine | PyTorch |
|---|---|
| scalar node | tensor |
| `requires_grad` concept | `requires_grad=True` |
| parents | computation graph edges |
| local backward function | autograd Function backward rule |
| `.grad` | `.grad` |
| `backward()` | `Tensor.backward()` |

## 11. Common failure modes

### Forgetting gradient accumulation

Shared branches produce incorrect gradients if contributions overwrite each other.

### Calling backward repeatedly without clearing gradients

PyTorch accumulates gradients by default.

### Detaching a value

Operations such as `.detach()` intentionally break gradient tracking.

### In-place mutation

Some in-place operations can invalidate values autograd needs for backward.

### Confusing numerical and analytical gradients

Numerical gradients depend on step size and floating-point precision; they are a test oracle, not the training mechanism.

## 12. Engineering rules

From this module onward:

1. test gradients on simple functions before trusting complex training
2. distinguish graph construction from gradient propagation
3. remember that gradients accumulate
4. clear optimizer gradients explicitly
5. use numerical gradient checks for custom math when practical
6. treat NaN/Inf gradients as defects requiring investigation

## 13. Exercises

1. Add division support.
2. Add an `exp()` node.
3. Add `tanh()`.
4. Verify $d(x^3)/dx=3x^2$ numerically.
5. Build $L=(xy+z)^2$ and derive each gradient manually.
6. Demonstrate PyTorch gradient accumulation across two backward calls.
7. Compare central-difference errors for several values of $h$.

## Next

Module 05 covers dataset splits, leakage, metrics and reproducibility—the experimental discipline required before training real vision models.
