# Episode 06 — Linear Classifiers for Images

Suggested title:

**Your First Trainable Vision Classifier — From Math to C++ Deployment**

Suggested thumbnail:

**LOGIT → PROBABILITY → DECISION**

Target duration: 45–55 minutes.

## Chapters

```text
00:00 Why linear classifiers still matter
05:00 Decision boundaries and logits
11:00 Sigmoid
16:00 Binary cross-entropy
23:00 Deriving gradients
29:00 Training from scratch
35:00 PyTorch nn.Linear equivalence
40:00 Portable model artifact
45:00 C++17 inference
50:00 Python ↔ C++ parity
```

## Demonstration

```bash
python 06_LinearClassifiers/python/linear_classifier_walkthrough.py
pytest 06_LinearClassifiers/tests -q

cmake -S 06_LinearClassifiers -B build/module06
cmake --build build/module06
ctest --test-dir build/module06 --output-on-failure

python 06_LinearClassifiers/python/linear_classifier_native_parity.py \
  build/module06/cv9x_linear_classifier \
  06_LinearClassifiers/artifacts/linear_binary_model.txt
```

## Core message

A linear classifier is the smallest complete example of learning parameters, serializing them, and reproducing the same inference result in a native runtime.
