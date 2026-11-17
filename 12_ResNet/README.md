# Module 12 — ResNet and Residual Learning

Very deep networks are not automatically easier to optimize. ResNet changed modern vision by turning a layer stack from “learn the whole transformation” into “learn a correction to an identity path.”

The central equation is:

$$
\mathbf{y}=\phi\left(F(\mathbf{x})+\mathbf{x}\right)
$$

where $F$ is the residual branch and the skip path carries the identity.

## Learning objectives

You should be able to:

- explain the degradation problem in deep networks
- distinguish residual learning from ordinary stacked layers
- derive the identity-skip equation
- explain why dimensions must match for addition
- understand projection shortcuts
- implement a residual block in PyTorch
- reproduce a small residual block in C++17
- verify Python/PyTorch/C++ numerical parity
- connect basic blocks, bottlenecks, and modern ResNet variants
- reason about residual blocks in production inference

## 1. Why residual learning?

Suppose the desired mapping is $H(\mathbf{x})$.

A conventional stack tries to learn $H(\mathbf{x})$ directly. A residual block instead learns:

$$
F(\mathbf{x}) = H(\mathbf{x})-\mathbf{x}
$$

so:

$$
H(\mathbf{x})=F(\mathbf{x})+\mathbf{x}
$$

If the identity mapping is already useful, the residual branch can learn a small correction rather than reconstructing identity from scratch.

## 2. Tiny residual block

This module uses a 1×1 convolution so the tensor arithmetic remains visible:

```text
input [C=2,H=2,W=2]
      │
      ├──────────── identity ────────────┐
      │                                  │
      └→ 1×1 convolution → F(x) ─────────┤
                                         ↓
                                   elementwise add
                                         ↓
                                        ReLU
```

At each pixel:

$$
F_o=b_o+\sum_i W_{o,i}x_i
$$

then:

$$
y_o=\operatorname{ReLU}(F_o+x_o)
$$

## 3. Shape compatibility and projection shortcuts

Identity addition requires compatible shapes. If channel count or spatial resolution changes, the shortcut may need a learned projection:

$$
\mathbf{y}=F(\mathbf{x})+W_s\mathbf{x}
$$

A 1×1 convolution is commonly used for the projection path.

## 4. Gradient intuition

For:

$$
\mathbf{y}=F(\mathbf{x})+\mathbf{x}
$$

the derivative contains an identity term:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
=
\frac{\partial F}{\partial \mathbf{x}}+I
$$

This creates a direct signal path through the block.

## 5. Basic block, bottleneck, and activation ordering

A basic block is conceptually:

```text
3×3 conv → activation → 3×3 conv → add skip
```

A bottleneck block reduces expensive computation:

```text
1×1 reduce → 3×3 spatial → 1×1 expand → add skip
```

Original and pre-activation ResNets differ in normalization/activation ordering. That ordering is part of the model contract.

## 6. Python, PyTorch, and C++17

The module provides:

- a manual tensor-level residual calculation
- a PyTorch `Conv2d` residual block
- a language-neutral artifact
- a C++17 implementation of the same 1×1 residual block
- complete output-tensor parity

## 7. Deployment implications

Residual networks introduce practical concerns:

- skip tensors must remain alive until addition
- layouts and shapes must match
- projection shortcuts add real computation
- graph fusion depends on backend support
- memory lifetime can matter as much as FLOPs

## 8. Failure modes

- incompatible skip/residual shapes
- channel-order mismatch
- missing projection for changed dimensions
- activation in the wrong location
- assuming deeper automatically means better
- checking only final class labels

## 9. Engineering rules

1. write skip-path shape explicitly
2. verify addition compatibility
3. preserve exact activation ordering
4. treat projection shortcuts as real computation
5. compare block tensors before full-model outputs
6. reason about skip-tensor memory lifetime
7. keep identity semantics visible in native code

## 10. Autonomous-perception relevance

Residual backbones remain common in detection, segmentation, and fusion systems:

```text
camera frame
   ↓
residual backbone
   ↓
multi-scale features
   ↓
detection / BEV / fusion
```

## 11. Book takeaway

> **A residual block gives the identity mapping explicitly and asks the learned branch to model the correction.**

## 12. Exercises

1. Add a second residual layer.
2. Implement a projection shortcut.
3. Extend the native block to 3×3 convolution.
4. Compare pre-activation and post-activation ordering.
5. Measure skip-tensor memory lifetime.
6. Explain why a zero residual branch still produces a meaningful result.

## Next

Module 13 studies efficient CNN building blocks and depthwise separable convolution.
