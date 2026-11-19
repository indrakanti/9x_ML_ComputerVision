# Module 15 — Hierarchical and Windowed Transformers

Global self-attention scales quadratically with token count. Windowed transformers reduce that cost by restricting attention to local token groups and building hierarchy across stages.

## Learning objectives

You should be able to:

- explain why global attention becomes expensive at high resolution
- partition a 2-D token grid into non-overlapping windows
- perform local scaled dot-product attention per window
- understand shifted-window motivation
- explain hierarchical feature maps and patch merging
- reproduce window partitioning and attention in C++17
- compare complete token outputs across Python/PyTorch/C++

## 1. From global to local attention

For $N$ tokens, global attention stores roughly:

$$
O(N^2)
$$

attention scores.

If tokens are split into windows of $M$ tokens, local attention becomes approximately:

$$
O\left(\frac{N}{M}M^2\right)=O(NM)
$$

when window size remains fixed.

## 2. Window partition

For a 4×4 token grid with a 2×2 window:

```text
a b | c d
e f | g h
----+----
i j | k l
m n | o p
```

the windows are:

```text
[a b e f]
[c d g h]
[i j m n]
[k l o p]
```

Ordering is part of the runtime contract.

## 3. Shifted windows

A fixed partition isolates neighboring windows. Shifted-window designs alternate the partition so information crosses previous boundaries.

The executable reference in this module focuses on non-shifted windows; the README explains the shifted extension.

## 4. Hierarchy

Hierarchical transformers progressively reduce spatial resolution and increase channel dimension, producing multi-scale features much like CNN backbones.

This is important for detection and segmentation, where multiple feature scales are useful.

## 5. Executable path

The module uses:

```text
4×4 token grid
embedding dimension = 2
window = 2×2
single-head local attention
```

Each window is processed independently with the same Q/K/V weights.

## 6. Deployment implications

Windowed attention trades global interaction for bounded local compute.

Important runtime variables include:

- window size
- number of windows
- token layout
- partition/merge copies
- Q/K/V buffer reuse
- softmax cost
- shifted-window data movement

## 7. Failure modes

- inconsistent window traversal order
- incorrect reshape/transpose
- attention leaking across windows
- wrong softmax axis
- forgetting positional/window metadata
- assuming lower asymptotic cost guarantees lower latency

## 8. Engineering rules

1. define token-grid layout
2. define window traversal order
3. keep windows independent unless explicitly shifted
4. compare complete token tensors
5. separate algorithmic complexity from measured latency
6. treat partition/merge memory movement as runtime cost

## 9. AV relevance

High-resolution multi-camera perception strongly benefits from architectures that control token growth.

Windowed/hierarchical backbones are one practical way to retain transformer modeling while bounding attention cost.

## 10. Book takeaway

> **Windowed attention controls quadratic cost by constraining who can attend to whom; hierarchy then restores multi-scale structure for vision tasks.**

## Next

Module 16 combines convolutional and transformer-style processing in one hybrid backbone.
