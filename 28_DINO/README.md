# Module 28 — DINO-Style Self-Supervised Learning

DINO-style learning trains a student network to match a slowly updated teacher network across different views of the same image, without class labels.

A compact view is:

```text
view A ─→ student ─→ student distribution
view B ─→ teacher ─→ centered/sharpened teacher distribution
                           ↓
                    cross-view loss
                           ↓
               EMA update of teacher
```

## Learning objectives

You should be able to:

- explain student/teacher self-distillation
- distinguish student and teacher temperatures
- understand teacher centering
- compute cross-entropy between teacher and student distributions
- explain why teacher parameters are updated by EMA
- reproduce DINO-style distribution/loss math in C++17
- reason about collapse avoidance and multi-view training

## 1. Student and teacher distributions

Let student logits be $z_s$ and teacher logits be $z_t$.

Student probabilities:

$$
p_s = \operatorname{softmax}(z_s / \tau_s)
$$

Teacher probabilities:

$$
p_t = \operatorname{softmax}((z_t-c)/\tau_t)
$$

where $c$ is the teacher center.

## 2. Why different temperatures?

A smaller teacher temperature sharpens the teacher distribution.

A larger student temperature makes the student prediction smoother.

The exact temperatures are part of the training configuration and should be checkpointed.

## 3. Centering

Without centering, some output dimensions can dominate.

The teacher center is typically an exponential moving average of teacher logits:

$$
c \leftarrow m_c c + (1-m_c)\bar{z}_t
$$

Centering is a training-state variable, not a deployment parameter.

## 4. Cross-view loss

For different augmented views $a$ and $b$:

$$
L_{a\rightarrow b}
=
-\sum_i p_t^{(b)}(i)\log p_s^{(a)}(i)
$$

The symmetric multi-view objective averages valid cross-view pairs.

This module uses two views and computes:

- student view A vs teacher view B
- student view B vs teacher view A

## 5. EMA teacher

Teacher parameters are not optimized directly with gradient descent.

Instead:

$$
\theta_t
\leftarrow
m\theta_t+(1-m)\theta_s
$$

A high momentum makes the teacher a slowly moving target.

## 6. Collapse

Self-supervised systems must avoid trivial solutions where every input receives the same representation.

DINO-style methods combine several mechanisms:

- teacher/student asymmetry
- centering
- sharpening
- EMA teacher
- multiple augmentations/crops

No single mechanism should be treated as a magic anti-collapse switch.

## 7. Executable reference

The artifact contains:

- student logits for two views
- teacher logits for two views
- teacher center
- student temperature
- teacher temperature
- student/teacher parameter vectors
- teacher EMA momentum

Python computes:

- student probabilities
- teacher centered/sharpened probabilities
- symmetric cross-view loss
- EMA-updated teacher parameters

PyTorch verifies the loss.

C++17 reproduces all numerical outputs.

## 8. Deployment implications

DINO is primarily a **representation-pretraining strategy**.

The downstream deployment typically uses the learned encoder:

```text
self-supervised pretraining
        ↓
foundation visual encoder
        ↓
probe / fine-tune / adapter
        ↓
production model
```

Teacher network, center state, augmentation pipeline, and DINO head can be training-only.

## 9. Failure modes

- applying teacher and student temperatures backwards
- forgetting teacher centering
- backpropagating through the teacher
- updating teacher with optimizer steps
- using same-view pairs only
- losing center state on resume
- changing EMA momentum without recording it

## 10. Engineering rules

1. checkpoint center state
2. checkpoint EMA teacher state
3. version temperatures and momentum
4. stop gradients through the teacher
5. keep augmentation/view policy reproducible
6. compare distributions and loss, not just final embeddings

## 11. AV relevance

Self-supervised training can exploit enormous unlabeled driving datasets and can produce transferable features for detection, segmentation, depth, and tracking.

## 12. Book takeaway

> **DINO-style learning creates labels from a slowly moving teacher. The training state therefore includes not only model weights, but also teacher momentum, temperature, centering, and view policy.**

## Reproducible commands

```bash
python 28_DINO/python/dino_walkthrough.py
pytest 28_DINO/tests
cmake -S 28_DINO -B build-m28 -G Ninja -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m28
ctest --test-dir build-m28 --output-on-failure
python 28_DINO/python/dino_native_parity.py \
  build-m28/cv9x_dino_math \
  28_DINO/artifacts/dino.txt
```

## Next

Module 29 studies reusable visual foundation backbones and task adapters.
