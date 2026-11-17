# Episode 10 — Regularization, Augmentation, Schedulers & Checkpointing

Suggested title:

**Why Training Loops Fail in Production — Regularization, Augmentation & Checkpoints**

Suggested thumbnail:

**LATEST ≠ BEST**

Target duration: 55–65 minutes.

## Chapters

```text
00:00 Overfitting
07:00 L2 and weight decay
14:00 Dropout: training vs inference
21:00 Deterministic augmentation
30:00 Learning-rate scheduling
38:00 Latest vs best checkpoint
46:00 Correct resume semantics
53:00 Deployment manifest
59:00 C++17 manifest validation
63:00 Python ↔ C++ parity
```

## Core message

Robust training is a reproducible state machine, not just repeated calls to optimizer.step().
