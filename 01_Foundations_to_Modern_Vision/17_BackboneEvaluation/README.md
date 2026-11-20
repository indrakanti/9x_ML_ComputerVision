# Module 17 — Backbone Evaluation and Engineering Trade-Space

A backbone should not be selected from top-1 accuracy alone. Production perception requires a multi-dimensional scorecard.

## Learning objectives

You should be able to:

- distinguish model quality from deployment quality
- compare parameter count, MAC proxy, latency, memory, and accuracy
- normalize metrics without hiding units
- build a repeatable scorecard
- explain why ranking changes with constraints
- encode the same score calculation in Python and C++17
- connect backbone selection to multi-camera AV compute budgets

## 1. Evaluation dimensions

A practical backbone review may include:

- task metric / accuracy proxy
- parameters
- MACs or FLOPs
- measured latency
- peak memory
- model size
- accelerator support
- export/runtime maturity
- determinism and reproducibility

## 2. Constraint-first thinking

For one camera, a 10 ms backbone may be acceptable.

For eight cameras at 30 FPS, the system-level budget changes dramatically.

Therefore the question is not:

> Which backbone is best?

It is:

> Which backbone satisfies this system's accuracy, latency, memory, and deployment constraints?

## 3. Scorecard in this module

The executable reference computes a weighted score from normalized:

- accuracy
- latency
- parameter count
- native-runtime readiness

Higher accuracy/readiness help the score. Higher latency/parameters hurt it.

The raw metrics remain visible so a score never replaces engineering judgment.

## 4. C++17 counterpart

The native tool reads the same CSV-like artifact and computes the same ranking.

Parity compares the ordered backbone names and scores.

## 5. Failure modes

- mixing measured and estimated latency
- comparing latency on different hardware without labeling it
- hiding normalization ranges
- collapsing all engineering judgment into one number
- ignoring batch size
- ignoring precision
- ignoring camera multiplicity

## 6. Engineering rules

1. preserve raw units
2. label hardware and precision
3. separate measured vs estimated metrics
4. document normalization
5. make weights explicit
6. treat score as decision support, not truth
7. include deployment readiness

## 7. AV relevance

Backbone selection directly affects how many camera streams fit within a compute and power envelope.

The capstone later uses the same style of trade-space analysis for surround perception.

## 8. Book takeaway

> **Architecture selection is a systems decision. Accuracy matters, but so do latency, memory, parallel camera count, runtime maturity, and evidence quality.**

## Stage 1.5 applied lab

The deterministic module above remains the mechanics reference. For a real-data run, continue with:

- [Applied Lab 01 — CIFAR-10 Backbones](../../applied_labs/01_CIFAR10Backbones/README.md)

That lab trains actual CIFAR-10 models, records held-out accuracy, writes checkpoints/plots, and keeps the dataset download optional.

## Next

Module 18 moves from backbones into detection fundamentals: boxes, anchors, IoU, and mAP.
