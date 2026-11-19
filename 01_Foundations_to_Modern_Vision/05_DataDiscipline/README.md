# Module 05 — Dataset Splits, Leakage, Metrics & Reproducibility

A model can be mathematically correct, implemented correctly, and still produce a misleading result if the experiment is designed badly.

This module establishes the **experimental contract** for the rest of the course.

The central idea is:

> A metric is only meaningful when the data split, provenance, randomness, preprocessing, and evaluation procedure are controlled.

---

## Learning objectives

By the end of this module you should be able to:

- explain the purpose of train, validation, and test sets
- distinguish model fitting from model selection and final evaluation
- identify common forms of dataset leakage
- build a deterministic split from stable sample identifiers
- explain why random seeds alone do not guarantee reproducibility
- compute binary classification confusion-matrix metrics
- understand when accuracy is misleading
- interpret precision, recall, specificity, and F1
- detect sample overlap between splits
- reproduce the same split and metric results in Python and C++17
- define the minimum provenance needed to make an experiment auditable

---

## 1. The experimental contract

Machine learning is not only:

```text
model + optimizer + data
```

A reproducible experiment is closer to:

```text
dataset revision
    +
sample identities
    +
split policy
    +
preprocessing
    +
random seeds
    +
model/configuration
    +
software environment
    +
evaluation definition
    =
experiment result
```

If any of those change silently, two runs that appear to be “the same experiment” may not actually be comparable.

---

## 2. Train, validation, and test sets

A standard split has three roles.

| Split | Purpose | May influence model design? |
|---|---|---|
| Train | Fit model parameters | Yes |
| Validation | Select hyperparameters, thresholds, checkpoints | Yes |
| Test | Final unbiased estimate | No |

The key rule is:

> The test set must not become part of the design loop.

If test performance repeatedly influences model choices, the test set has effectively become another validation set.

### Typical flow

```text
dataset
   │
   ├── train ──────> fit parameters
   │
   ├── validation -> select configuration
   │
   └── test ───────> final evaluation only
```

---

## 3. What is data leakage?

Leakage occurs when information that should be unavailable during training or model selection influences the result.

Leakage can make a weak system appear excellent.

### 3.1 Exact sample duplication

The same image appears in both train and validation/test.

```text
train: image_0042.jpg
test : image_0042.jpg   ← leakage
```

### 3.2 Near-duplicate leakage

Different files represent effectively the same scene.

Examples:

- adjacent video frames
- burst-mode camera captures
- crops from the same original image
- augmented copies split across partitions

For video datasets, frame-level random splitting is particularly dangerous.

### 3.3 Subject or identity leakage

Images of the same person, patient, vehicle, route, or physical object appear across train and test when the intended generalization target is unseen identity.

### 3.4 Temporal leakage

Future information becomes available to a model that is supposed to predict the future.

### 3.5 Preprocessing leakage

Statistics are computed using all data before splitting.

Bad:

```text
entire dataset
     ↓
compute normalization statistics
     ↓
split train / validation / test
```

Better:

```text
split first
   ↓
fit preprocessing on train
   ↓
apply frozen preprocessing to validation/test
```

---

## 4. Deterministic splitting by stable identity

A common beginner approach is:

```python
shuffle(samples)
split(samples)
```

That may be reproducible for one exact list, but assignments can change if:

- a sample is inserted
- ordering changes
- filesystem traversal changes
- the random-number implementation changes

A stronger production pattern is to derive the split from a **stable sample identifier**.

This module uses a stable FNV-1a hash:

```text
sample_id
   ↓
stable 64-bit hash
   ↓
bucket in [0, 9999]
   ↓
train / validation / test
```

For a 70/15/15 split:

```text
0–6999     train
7000–8499  validation
8500–9999  test
```

The important property is:

> The split assignment is a function of sample identity, not list ordering.

That means Python and C++ can independently reproduce the same assignment.

---

## 5. Stratification and grouping

Stable hashing alone does not guarantee class balance.

For classification, you may also need **stratification** so class proportions remain reasonably similar across splits.

For grouped data, the split unit must often be the group rather than the sample.

Examples:

| Problem | Correct split unit |
|---|---|
| Video classification | video/sequence |
| Medical imaging | patient |
| Driver monitoring | person |
| Autonomous-driving perception | route/log/scene |
| Product recognition | physical item/session |

A useful rule is:

> Split at the level of independence you want the model to generalize across.

---

## 6. Confusion matrix

For binary classification:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

These four counts define many important metrics.

### Accuracy

$$
\text{Accuracy}
=
\frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

$$
\text{Precision}
=
\frac{TP}{TP+FP}
$$

Precision asks:

> When the model predicts positive, how often is it correct?

### Recall / sensitivity

$$
\text{Recall}
=
\frac{TP}{TP+FN}
$$

Recall asks:

> Of the true positive cases, how many did the model find?

### Specificity

$$
\text{Specificity}
=
\frac{TN}{TN+FP}
$$

### F1 score

$$
F_1
=
2
\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

---

## 7. Why accuracy can be misleading

Suppose only 1% of images contain the event of interest.

A classifier that always predicts “negative” gets:

$$
99\% \text{ accuracy}
$$

but:

$$
\text{Recall}=0
$$

For safety-relevant or rare-event vision tasks, this can be catastrophic.

Always ask:

- What is the class distribution?
- What error type matters most?
- Is false negative cost equal to false positive cost?
- Is the metric threshold-dependent?
- Is the reported result statistically stable?

---

## 8. Reproducibility is more than a random seed

A seed controls only one source of variation.

A reproducibility record should include at least:

```text
dataset revision
sample/group identifiers
split algorithm + parameters
random seeds
preprocessing configuration
augmentation configuration
model architecture/version
optimizer + hyperparameters
checkpoint selection rule
software versions
device/precision
evaluation code version
metric definition
```

Later modules will turn this into experiment configuration and artifact metadata.

---

## 9. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Python formatting / lint | Required | Dataset and metric reference implementation |
| mypy | Required | Typed split/metric helpers |
| pytest | Required | Leakage, split stability, and metrics |
| CPU PyTorch tests | N/A | No tensor/model behavior is needed here |
| C++ configure/build/CTest | Required | Native split and metric behavior must be reproducible |
| Python ↔ C++ numerical parity | Required | Split assignments and metrics must match |
| ONNX tests | N/A | No model artifact exists yet |
| GPU | N/A | No accelerated computation is introduced |

---

## 10. Module structure

```text
05_DataDiscipline/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── python/
│   ├── data_discipline.py
│   ├── data_discipline_walkthrough.py
│   └── data_discipline_native_parity.py
├── cpp/
│   └── data_discipline.cpp
└── tests/
    ├── conftest.py
    └── test_data_discipline.py
```

---

## 11. Python implementation

The Python reference provides:

- FNV-1a 64-bit stable hashing
- deterministic 70/15/15 sample assignment
- duplicate/leakage detection across named splits
- binary confusion-matrix accumulation
- accuracy
- precision
- recall
- specificity
- F1

The functions are deliberately small and inspectable.

This matters because evaluation code should be easier to audit than model code.

---

## 12. C++17 native counterpart

The native implementation reproduces:

- the same FNV-1a hash
- the same split bucket policy
- the same binary confusion matrix
- the same metric definitions

The executable supports:

```bash
cv9x_data_discipline --self-test
cv9x_data_discipline --json
```

The JSON mode is used as a language-neutral evidence boundary.

Python invokes the compiled C++ executable and verifies exact split assignment plus numerical metric parity.

### Why this matters

Production systems frequently compute evaluation, monitoring, or runtime telemetry outside the Python training process.

A metric definition must not silently change because the language changes.

The proof is:

```text
same sample IDs / labels
        │
        ├── Python
        │     ↓
        │  reference result
        │
        └── C++17
              ↓
           native result

        compare
          ↓
        PASS
```

---

## 13. Run the module

### Python tests

```bash
source .venv/bin/activate

python 05_DataDiscipline/python/data_discipline_walkthrough.py
pytest 05_DataDiscipline/tests -q
```

### Standalone C++ build

```bash
cmake -S 05_DataDiscipline -B build/module05 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON

cmake --build build/module05 --parallel
ctest --test-dir build/module05 --output-on-failure
```

### Python ↔ C++ parity

```bash
python 05_DataDiscipline/python/data_discipline_native_parity.py \
  build/module05/cv9x_data_discipline
```

### Full repository gate

```bash
pytest

cmake -S . -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON

cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
```

---

## 14. Worked example

Assume:

```text
truth       = [1, 1, 1, 0, 0, 0]
prediction  = [1, 1, 0, 1, 0, 0]
```

Then:

```text
TP = 2
FN = 1
FP = 1
TN = 2
```

So:

$$
\text{Accuracy}=\frac{4}{6}=0.6667
$$

$$
\text{Precision}=\frac{2}{3}=0.6667
$$

$$
\text{Recall}=\frac{2}{3}=0.6667
$$

$$
F_1=0.6667
$$

The point is not the particular number.

The point is that the metric has an explicit definition that can be reproduced in Python, C++, CI, and later production telemetry.

---

## 15. Failure modes

### Randomly splitting video frames

Adjacent frames leak scene information between train and test.

### Computing normalization globally

Validation/test statistics influence preprocessing.

### Reusing the test set during development

Repeated test inspection turns it into a validation set.

### Reporting only accuracy

Rare-class failures become invisible.

### Changing split logic between experiments

Model comparisons become invalid.

### Using file order as identity

Filesystem behavior can silently alter partitions.

### Ignoring zero denominators

Metrics can produce NaN or misleading values when no positive predictions or examples exist.

This module defines zero-denominator precision/recall/F1 as 0.0 so behavior is explicit and deterministic.

---

## 16. Engineering rules

From this module onward:

1. every sample must have a stable identity
2. split policy must be version-controlled
3. group boundaries must be respected
4. train/validation/test overlap is a test failure
5. preprocessing is fit on training data only
6. metric definitions must be explicit
7. class counts accompany aggregate metrics
8. experiment seeds and dataset revisions are recorded
9. Python and native metric behavior must agree where both exist
10. test data must not become part of the model-design loop

---

## 17. Book takeaway

A machine-learning result is not just a number.

A trustworthy result is:

> **a metric + a dataset definition + a split contract + provenance + reproducible evaluation code.**

This principle becomes increasingly important as the course moves into larger models, more expensive experiments, and production deployment.

---

## 18. Exercises

1. Change the split ratio to 80/10/10 while preserving deterministic assignment.
2. Add a grouped split keyed by `sequence_id`.
3. Construct a leakage example using adjacent video frames.
4. Add balanced accuracy.
5. Add Matthews correlation coefficient.
6. Compare accuracy and recall on a 99:1 imbalanced dataset.
7. Record a split manifest to JSON.
8. Verify that adding a new sample does not move existing samples between splits.
9. Create a deliberate Python/C++ metric mismatch and confirm parity CI detects it.
10. Design a split strategy for autonomous-driving logs where multiple cameras share the same scene.

## Next

Module 06 begins learned decision boundaries with linear classifiers for images.
