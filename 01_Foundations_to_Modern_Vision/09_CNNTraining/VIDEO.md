# Episode 09 — CNN Training Loop from Scratch

Suggested title:

**Train Your First CNN Properly — Forward, Backward, Optimizer, Checkpoint, C++**

Suggested thumbnail:

**FORWARD → LOSS → BACKWARD → STEP**

Target duration: 55–65 minutes.

## Chapters

```text
00:00 From convolution to a trained CNN
06:00 Tiny synthetic vision task
12:00 CNN architecture
18:00 Cross-entropy and logits
24:00 zero_grad / forward / backward / step
33:00 Epochs and evaluation
40:00 train() vs eval() vs no_grad()
46:00 Checkpoints vs inference artifacts
53:00 Exporting learned CNN parameters
58:00 C++17 forward-only runtime
63:00 Python ↔ C++ parity
```

## Core message

Training is stateful optimization; deployment is forward-only inference. A production ML workflow makes the boundary explicit and reproducible.
