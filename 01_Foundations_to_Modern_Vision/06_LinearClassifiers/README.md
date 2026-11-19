# Module 06 — Linear Classifiers for Images

Linear classifiers are the first place where the course moves from **measuring data** to **learning a decision boundary**.

They are simple enough to derive completely, yet important enough to reappear throughout modern deep learning.

A neural-network classification head is still fundamentally computing something like:

$$
z = \mathbf{w}^{T}\mathbf{x} + b
$$

followed by a probability or score transformation.

This module builds that idea from first principles, trains it in Python and PyTorch, and then deploys the same learned decision function in C++17 using a small language-neutral model artifact.

---

## Learning objectives

By the end of this module you should be able to:

- explain a linear decision boundary geometrically
- compute logits from weights, features, and bias
- explain why logits are not probabilities
- derive the sigmoid function for binary classification
- explain binary cross-entropy from a probabilistic perspective
- train a binary linear classifier with gradient descent
- compare a from-scratch implementation against `torch.nn.Linear`
- serialize a minimal model artifact
- load the same model artifact in Python and C++17
- prove Python ↔ C++ inference parity
- understand why this pattern becomes the basis for later detector classification heads

---

## 1. From features to a decision

Suppose an image has already been converted into a feature vector:

$$
\mathbf{x}
=
[x_1,x_2,\ldots,x_d]^T
$$

A linear classifier computes:

$$
z = \mathbf{w}^{T}\mathbf{x}+b
$$

where:

- $\mathbf{w}$ contains learned feature weights
- $b$ is a bias term
- $z$ is the **logit**

For two dimensions:

$$
z=w_1x_1+w_2x_2+b
$$

The decision boundary is:

$$
w_1x_1+w_2x_2+b=0
$$

That equation describes a line in 2-D, a plane in 3-D, and a hyperplane in higher-dimensional feature spaces.

## 2. Geometric intuition

```text
feature vector x
      ↓
projection onto w
      ↓
signed score
      ↓
logit z
```

If $z$ is strongly positive, the classifier favors the positive class. If it is strongly negative, it favors the negative class. If $z$ is close to zero, the sample lies near the decision boundary.

## 3. From logit to probability

The sigmoid function maps a real-valued logit to $(0,1)$:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore:

$$
p(y=1|\mathbf{x})=\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

A probability threshold such as $0.5$ gives:

```text
p >= 0.5  → positive
p <  0.5  → negative
```

For sigmoid, $p=0.5$ corresponds exactly to $z=0$.

## 4. Binary cross-entropy

For target $y\in\{0,1\}$ and predicted probability $p$:

$$
L=-y\log p-(1-y)\log(1-p)
$$

A numerically stable logits form is:

$$
L(z,y)=\max(z,0)-zy+\log(1+e^{-|z|})
$$

Production frameworks prefer this style because it avoids overflow for large-magnitude logits.

## 5. Gradient of logistic regression

For one sample:

$$
p=\sigma(z)
$$

and:

$$
\frac{\partial L}{\partial z}=p-y
$$

Therefore:

$$
\frac{\partial L}{\partial \mathbf{w}}=(p-y)\mathbf{x}
$$

and:

$$
\frac{\partial L}{\partial b}=p-y
$$

For a batch, the module averages these gradients and applies gradient descent.

## 6. Tiny image-feature example

To keep the math visible, each example is represented by four normalized image-like features:

```text
[brightness,
 edge_energy,
 vertical_structure,
 horizontal_structure]
```

Later CNNs will learn features automatically, but a final classification head still converts a learned feature vector into logits.

## 7. From-scratch Python implementation

The Python reference implements:

- linear logits
- stable sigmoid
- BCE-with-logits
- probability prediction
- thresholded prediction
- batch gradients
- gradient-descent training
- model save/load

The loop is explicit:

```text
features
   ↓
logits
   ↓
probabilities
   ↓
loss
   ↓
gradients
   ↓
weight update
```

## 8. PyTorch implementation

PyTorch expresses the same classifier as:

```python
torch.nn.Linear(input_features, 1)
```

The tests load identical weights and bias into `torch.nn.Linear` and compare its logit against the hand-written implementation.

> Framework convenience should not hide the mathematical operation being executed.

## 9. Minimal model artifact

This module introduces a small portable model artifact before ONNX:

```text
version=1
input_size=4
weights=0.8,-0.4,1.1,0.6
bias=-0.25
threshold=0.5
```

The purpose is to teach the deployment contract:

```text
model parameters
      ↓
serialized artifact
      ↓
Python loader
      ↓
C++ loader
      ↓
same inference result
```

Later, ONNX replaces this toy format for real neural networks.

## 10. C++17 native counterpart

The C++ implementation:

- parses the same artifact
- validates input dimension
- computes the dot product
- adds bias
- applies sigmoid
- applies threshold
- exposes deterministic JSON output
- includes a native self-test

## 11. Python ↔ C++ parity

Parity checks compare:

- logit
- probability
- predicted class

This is the first module where the course proves a real **model parameter artifact** outside the training language.

## 12. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Python formatting / lint | Required | From-scratch training and artifact loader |
| mypy | Required | Typed model and training functions |
| pytest | Required | Math, training, artifact, and edge cases |
| CPU PyTorch tests | Required | Hand-written linear math is checked against PyTorch |
| C++ configure/build/CTest | Required | Native model loading and inference |
| Python ↔ C++ numerical parity | Required | Same model artifact must produce same result |
| ONNX | N/A | Reserved for the first neural-model export checkpoint |
| GPU | N/A | This classifier is too small for meaningful acceleration |

## 13. Module structure

```text
06_LinearClassifiers/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── artifacts/
│   └── linear_binary_model.txt
├── python/
│   ├── linear_classifier.py
│   ├── linear_classifier_walkthrough.py
│   └── linear_classifier_native_parity.py
├── cpp/
│   └── linear_classifier.cpp
└── tests/
    ├── conftest.py
    └── test_linear_classifier.py
```

## 14. Run the module

### Python

```bash
source .venv/bin/activate

python 06_LinearClassifiers/python/linear_classifier_walkthrough.py
pytest 06_LinearClassifiers/tests -q
```

### Native C++17

```bash
cmake -S 06_LinearClassifiers -B build/module06 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON

cmake --build build/module06 --parallel
ctest --test-dir build/module06 --output-on-failure
```

### Python ↔ C++ parity

```bash
python 06_LinearClassifiers/python/linear_classifier_native_parity.py \
  build/module06/cv9x_linear_classifier \
  06_LinearClassifiers/artifacts/linear_binary_model.txt
```

## 15. Worked example

Take:

$$
\mathbf{x}=[0.5,0.2,0.7,0.1]
$$

$$
\mathbf{w}=[0.8,-0.4,1.1,0.6]
$$

and $b=-0.25$.

Then:

$$
z
=
0.8(0.5)-0.4(0.2)+1.1(0.7)+0.6(0.1)-0.25
=
0.90
$$

So:

$$
p=\sigma(0.90)\approx0.711
$$

With threshold $0.5$, the predicted class is positive.

Python and C++ must reproduce the same result from the same artifact.

## 16. Failure modes

### Treating logits as probabilities

A logit can be any real number. It is not bounded to $[0,1]$.

### Applying sigmoid twice

If a framework loss already expects logits, applying sigmoid first changes optimization behavior and can reduce numerical stability.

### Ignoring feature scaling

A feature with a much larger numeric range can dominate optimization.

### Using accuracy alone

Module 05 already showed why a classifier can look strong while failing the important class.

### Changing the threshold silently

A threshold change changes operating behavior even when weights remain identical.

### Training/inference preprocessing mismatch

The model can be mathematically identical while behavior changes because normalization changed.

### Artifact drift

If C++ loads different weights, bias, threshold, or input dimensionality than Python, deployed behavior no longer matches the training reference.

## 17. Engineering rules

From this module onward:

1. distinguish logits from probabilities
2. keep preprocessing part of the model contract
3. record decision thresholds explicitly
4. use numerically stable loss implementations
5. prove framework math against a simple reference where practical
6. serialize model parameters instead of copying them into runtime source
7. validate input dimension before inference
8. compare Python and native outputs
9. version model artifacts
10. treat model configuration as deployable data, not source-code constants

## 18. Connection to surround object detection

Modern detectors contain classification heads that repeatedly make decisions from learned feature vectors:

```text
CNN / Transformer features
          ↓
classification head
          ↓
class logits
          ↓
scores / probabilities
```

The detector is much larger, but the concepts remain:

- weighted features
- logits
- losses
- thresholds
- calibration
- deployment equivalence

This module therefore becomes a foundation for the later **surround camera + LiDAR detection and tracking capstone**.

## 19. Book takeaway

A linear classifier is the smallest useful example of the complete ML deployment lifecycle:

> **learn parameters → serialize them → load them in another runtime → reproduce the same decision.**

The mathematics is simple, but the engineering contract is the same one required by much larger models.

## 20. Exercises

1. Add L2 regularization.
2. Change threshold and observe precision/recall tradeoffs.
3. Train with differently scaled features.
4. Extend to multiclass softmax.
5. Save a trained artifact instead of using the supplied example.
6. Add artifact checksum validation.
7. Compare float32 and float64 inference.
8. Add feature names to the artifact.
9. Introduce a Python/C++ artifact mismatch and verify parity fails.
10. Explain how this classifier could become a vehicle-detector head.

## Next

Module 07 introduces multilayer perceptrons, nonlinear activations, initialization, and normalization.
