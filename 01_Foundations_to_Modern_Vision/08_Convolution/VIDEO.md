# Episode 08 — Convolution from First Principles

Suggested title:

**How Convolution Really Works — Python, C++, and the GPU Execution Model**

Suggested thumbnail:

**KERNEL × IMAGE → FEATURE MAP**

Target duration: 55–65 minutes.

## Chapters

```text
00:00 Why convolution matters
06:00 Cross-correlation vs convolution
12:00 Output shape
18:00 Stride and padding
25:00 Multi-channel convolution
33:00 First-principles Python
40:00 PyTorch equivalence
46:00 C++17 convolution
52:00 Python ↔ C++ parity
57:00 CPU memory behavior
61:00 GPU copies, launch and synchronization
```

## Core message

Convolution math is straightforward; production engineering is preserving that math while mapping it efficiently onto memory and parallel hardware.
