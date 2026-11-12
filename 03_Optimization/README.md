# Module 03 — Optimization: Losses, Gradients, SGD, Momentum and Adam

This module explains how learning actually changes model parameters.

## Learning objectives

By the end of this module you should be able to:

- define an objective function
- distinguish loss, gradient and optimizer state
- derive gradient descent on a scalar quadratic
- explain batch gradient descent vs SGD
- implement momentum from first principles
- implement Adam from first principles
- explain learning rate, momentum and adaptive scaling
- compare manual updates with PyTorch optimizers
- recognize divergence, oscillation and vanishing update behavior
- understand why optimization behavior depends on parameter scaling

## 1. The optimization loop

Training repeatedly performs:

~~~text
data
 ↓
model
 ↓
prediction
 ↓
loss
 ↓
gradient
 ↓
optimizer
 ↓
updated parameters
~~~

The optimizer does not "understand" the task. It only receives gradients and updates parameters according to a rule.

## 2. Objective function

Suppose:

$
f(w)=(w-3)^2
$

The minimum is at:

$
w^*=3
$

The derivative is:

[
rac{df}{dw}=2(w-3)
]

Gradient descent updates:

[
w_{t+1}=w_t-eta 
abla f(w_t)
]

where $\eta$ is the learning rate.

## 3. Learning rate

If the learning rate is too small:

~~~text
stable
but slow
~~~

If it is too large:

~~~text
oscillation
divergence
numerical instability
~~~

A learning rate is not simply "speed." It changes the discrete dynamics of optimization.

## 4. SGD

For a dataset objective:

[
L(	heta)=rac{1}{N}sum_{i=1}^{N}ell_i(	heta)
]

full-batch gradient descent uses all samples.

Stochastic/minibatch gradient descent uses an estimate:

[
g_t approx 
abla L(	heta_t)
]

The noise is often useful because training becomes computationally practical and can escape some poorly conditioned behavior.

## 5. Momentum

Momentum maintains state:

[
v_t=eta v_{t-1}+g_t
]

and updates:

[
	heta_{t+1}=	heta_t-eta v_t
]

Intuition:

~~~text
gradient = local direction
momentum = accumulated direction
~~~

Momentum can reduce zig-zagging in directions where gradients repeatedly change sign.

## 6. Adam

Adam tracks first and second moments:

[
m_t=eta_1m_{t-1}+(1-eta_1)g_t
]

[
v_t=eta_2v_{t-1}+(1-eta_2)g_t^2
]

Bias-corrected estimates:

[
hat m_t=rac{m_t}{1-eta_1^t}
]

[
hat v_t=rac{v_t}{1-eta_2^t}
]

Update:

[
	heta_{t+1}
=
	heta_t
-
eta
rac{hat m_t}{sqrt{hat v_t}+epsilon}
]

Adam adapts update magnitude using recent first- and second-moment estimates.

## 7. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Python formatting / lint | Required | Python optimizer implementations |
| mypy | Required | Typed optimizer helpers |
| pytest | Required | Optimizer math and state validation |
| CPU PyTorch tests | Required | Manual SGD is checked against `torch.optim.SGD` |
| C++ configure/build/CTest | Required | Optimizer state/update math is implemented natively |
| Python ↔ C++ numerical parity | Required | SGD, Momentum and Adam deterministic outputs must match |
| ONNX tests | N/A | No exported model |
| Python ↔ exported-model equivalence | N/A | No exported model |

## 8. What this module implements

The module contains:

~~~text
03_Optimization/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── python/
│   ├── optimization.py
│   ├── optimization_walkthrough.py
│   └── native_parity.py
├── cpp/
│   └── optimizer_state.cpp
└── tests/
    ├── conftest.py
    └── test_optimization.py
~~~

The module implements the same optimizer-state mathematics in Python and C++17. ONNX remains N/A because there is no model artifact yet.

## 9. Run

~~~bash
source .venv/bin/activate

python 03_Optimization/python/optimization_walkthrough.py
pytest 03_Optimization/tests -q

cmake -S 03_Optimization -B build/module03 -DCMAKE_BUILD_TYPE=Release
cmake --build build/module03 --parallel
ctest --test-dir build/module03 --output-on-failure

python 03_Optimization/python/native_parity.py \
  build/module03/cv9x_optimizer_state
~~~

## 10. Native C++ counterpart

The C++17 implementation mirrors the first-principles Python optimizer state:

- gradient-descent update
- momentum velocity accumulation
- Adam first moment
- Adam second moment
- bias correction
- epsilon stabilization

The parity checker runs both implementations over the same deterministic quadratic objective and compares the resulting values.

## 11. Experiments

The walkthrough compares:

- plain gradient descent
- momentum
- Adam
- manual implementations
- PyTorch optimizer behavior

on deterministic objectives.

## 12. Failure modes

### Learning rate too large

Loss may oscillate or explode.

### Learning rate too small

Training appears stable but makes negligible progress.

### Gradient scale mismatch

Different parameters can receive gradients with very different magnitudes.

### Momentum overshoot

Momentum can carry parameters past a minimum if hyperparameters are poorly chosen.

### Adam interpreted as universally superior

Adam often converges quickly, but optimization speed and final generalization are different questions.

## 13. Engineering rules

From this module onward:

1. log loss over time
2. record optimizer type and hyperparameters
3. distinguish optimizer state from model parameters
4. make learning rate explicit
5. seed stochastic experiments
6. test optimizer math on simple objectives before trusting a training stack
7. treat NaN/Inf gradients as failures, not "training noise"

## 14. Exercises

1. Change the quadratic minimum from 3 to -5.
2. Compare learning rates 0.001, 0.1 and 1.1.
3. Add gradient clipping.
4. Plot optimization trajectories.
5. Compare SGD and momentum on an anisotropic 2-D quadratic.
6. Reproduce the Adam update by hand for three steps.
7. Compare manual Adam with `torch.optim.Adam`.

## Next

Module 04 derives automatic differentiation from scalar computation graphs and connects the chain rule directly to PyTorch autograd.
