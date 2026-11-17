# Module 13 — Efficient CNNs: Depthwise Separable Convolution

Efficient CNNs reduce computation by changing the structure of convolution rather than simply shrinking the input.

The key MobileNet-style idea is to factor a standard convolution into:

```text
depthwise spatial filtering
          ↓
pointwise 1×1 channel mixing
```

## Learning objectives

You should be able to:

- compute the cost of a standard convolution
- explain depthwise convolution
- explain pointwise convolution
- derive the MAC reduction of depthwise separable convolution
- implement depthwise + pointwise inference
- reproduce the same operator in C++17
- compare full output tensors across Python/PyTorch/C++
- explain why arithmetic reduction does not guarantee latency reduction
- connect efficient blocks to MobileNet/EfficientNet-style architectures

## 1. Standard convolution cost

Ignoring bias, standard convolution requires approximately:

$$
H_oW_oC_{out}C_{in}K_hK_w
$$

multiply-accumulate operations.

## 2. Depthwise convolution

Depthwise convolution applies one spatial kernel per input channel:

$$
Y_c=X_c*W_c
$$

Cost:

$$
H_oW_oC K_hK_w
$$

No channel mixing happens in this stage.

## 3. Pointwise convolution

A 1×1 convolution mixes channels:

$$
Z_o(i,j)=b_o+\sum_cP_{o,c}Y_c(i,j)
$$

Cost:

$$
H_oW_oC_{in}C_{out}
$$

## 4. Combined cost

Depthwise-separable cost is:

$$
H_oW_oC_{in}K_hK_w
+
H_oW_oC_{in}C_{out}
$$

This can be much smaller than a standard convolution.

## 5. Tiny executable example

```text
input        [2,3,3]
depthwise    2 kernels of 2×2
depth output [2,2,2]
pointwise    [2 output channels, 2 input channels]
final output [2,2,2]
```

PyTorch expresses depthwise convolution with `groups=channels`.

## 6. Python, PyTorch, and C++17

The module provides:

- a manual depthwise stage
- a manual pointwise stage
- a PyTorch grouped-convolution reference
- a C++17 equivalent
- complete output-tensor parity

## 7. MAC-count evidence

The code reports both:

- equivalent standard-convolution MACs
- depthwise-separable MACs

But MAC count is not latency. Real runtime also depends on memory bandwidth, vectorization, launch overhead, tensor layout, batch size, and backend optimization.

## 8. EfficientNet connection

EfficientNet combines efficient blocks with compound scaling across:

- depth
- width
- resolution

The broader lesson is that architecture efficiency is multi-dimensional.

## 9. Deployment implications

Depthwise convolution may use fewer FLOPs while achieving worse hardware utilization on some targets.

Therefore deployment evaluation should include:

```text
accuracy
+ arithmetic
+ latency
+ memory
+ backend support
```

## 10. Failure modes

- forgetting `groups=C`
- mixing channels in the depthwise stage
- wrong pointwise weight layout
- assuming lower MACs means lower latency
- comparing different output shapes
- ignoring memory bandwidth

## 11. Engineering rules

1. separate spatial filtering from channel mixing
2. write groups explicitly
3. compare equivalent output shapes
4. report MACs and measured latency separately
5. preserve tensor layout across runtimes
6. validate every output element
7. profile on the target accelerator

## 12. Autonomous-driving relevance

Efficient backbones matter when many cameras run concurrently:

```text
6–8 cameras
   ↓
efficient visual backbones
   ↓
shared BEV / fusion system
```

A per-camera saving multiplies across the complete surround stack.

## 13. Book takeaway

> **Depthwise separable convolution saves arithmetic by decoupling spatial filtering from channel mixing; real efficiency still depends on the hardware/runtime implementation.**

## 14. Exercises

1. Add stride 2.
2. Add ReLU6.
3. Compute the MAC ratio for 64→128 channels with a 3×3 kernel.
4. Implement an inverted residual block.
5. Benchmark standard vs separable convolution on CPU.
6. Explain why an accelerator might dislike small depthwise kernels.

## Next

Module 14 introduces Vision Transformers.
