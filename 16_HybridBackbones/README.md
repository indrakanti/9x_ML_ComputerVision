# Module 16 — Hybrid CNN–Transformer Backbones

Hybrid backbones combine convolutional spatial bias with transformer-style global interaction.

## Learning objectives

You should be able to:

- explain why CNN stems are useful before attention
- convert convolutional feature maps into tokens
- apply token attention after spatial feature extraction
- reason about hybrid memory/layout transitions
- reproduce the hybrid forward path in C++17
- compare complete token outputs across Python/PyTorch/C++

## 1. Hybrid motivation

CNNs provide locality and efficient early spatial processing. Transformers provide flexible token interaction.

A hybrid backbone can use:

```text
image
  ↓
convolutional stem
  ↓
feature map
  ↓
flatten to tokens
  ↓
self-attention
  ↓
hybrid representation
```

## 2. Executable reference

The module uses:

- 4×4 grayscale image
- one 2×2 convolutional filter with stride 2
- four spatial outputs
- embedding dimension 2 using a learned projection
- one self-attention layer

This creates a compact but real CNN→token→attention path.

## 3. Layout boundary

The important systems transition is:

```text
NCHW feature map
      ↓
flatten spatial positions
      ↓
[N, tokens, channels/embed]
```

Layout errors at this boundary are common and difficult to debug.

## 4. Deployment implications

Hybrid backbones may incur:

- convolution workspace
- feature-map to token reshaping/transposition
- Q/K/V buffers
- attention score matrices
- different kernel families in one graph

Runtime quality depends on how efficiently those boundaries are implemented.

## 5. Engineering rules

1. define feature-map layout
2. define spatial-to-token order
3. verify token count
4. preserve projection weight layout
5. compare complete token outputs
6. measure layout conversion cost separately

## 6. AV relevance

Hybrid backbones can be attractive for surround perception because early convolution handles image locality efficiently while later attention models longer-range context.

## 7. Book takeaway

> **Hybrid backbones are not “CNN plus transformer” conceptually; they are a carefully defined tensor-layout bridge between two different computational regimes.**

## Next

Module 17 evaluates backbone choices using an explicit engineering scorecard.
