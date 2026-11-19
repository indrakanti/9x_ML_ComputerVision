# Module 08 — Convolution from First Principles

Convolution is the operation that lets neural networks exploit image structure. This module builds 2-D convolution from explicit loops, verifies it against PyTorch, implements the same operation in C++17, and introduces the first serious CPU/GPU execution model in the course.

## Learning objectives

You should be able to:

- explain locality and weight sharing
- derive output dimensions from kernel, stride, and padding
- implement multi-channel 2-D cross-correlation from first principles
- understand CHW/NCHW tensor layouts
- compare hand-written results against `torch.nn.functional.conv2d`
- reproduce the operation in C++17
- understand host/device copies, GPU kernel launch, and synchronization
- separate numerical correctness from acceleration

## 1. Convolution in deep learning

Deep-learning libraries normally implement cross-correlation:

$$
Y(i,j)=b+\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}
X(iS_h+u-P_h,jS_w+v-P_w)W(u,v)
$$

The kernel is not flipped.

## 2. Output shape

$$
H_{out}=
\left\lfloor\frac{H+2P_h-K_h}{S_h}\right\rfloor+1
$$

$$
W_{out}=
\left\lfloor\frac{W+2P_w-K_w}{S_w}\right\rfloor+1
$$

Shape math is part of the runtime contract, not bookkeeping.

## 3. Why convolution helps vision

Convolution introduces:

- locality
- weight sharing
- translation structure

## 4. Multi-channel convolution

$$
Y_o(i,j)=b_o+
\sum_c\sum_u\sum_v
X_c(iS_h+u-P_h,jS_w+v-P_w)W_{o,c}(u,v)
$$

Tensor contracts:

```text
input   [C_in, H, W]
weight  [C_out, C_in, K_h, K_w]
bias    [C_out]
output  [C_out, H_out, W_out]
```

PyTorch batches images as NCHW:

```text
[N, C, H, W]
```

## 5. Stride and padding

Stride controls kernel movement. Padding controls border behavior and output size.

For a 3×3 kernel with stride 1, padding 1 preserves spatial dimensions.

## 6. First-principles Python

The reference loops explicitly over:

```text
output channel
  input channel
    output row
      output column
        kernel row
          kernel column
```

The implementation is deliberately unoptimized so every index remains visible.

## 7. PyTorch equivalence

The same input, weights, bias, stride, and padding are passed to:

```python
torch.nn.functional.conv2d
```

The full output tensor must match the reference numerically.

## 8. C++17 native counterpart

The C++ implementation uses the same conventions:

```text
input  [C,H,W]
weight [O,C,KH,KW]
output [O,OH,OW]
```

It supports stride, padding, multiple input/output channels, bias, JSON output, and CTest.

## 9. Python ↔ C++ parity

Every output element is compared:

```text
same tensors + same geometry
          │
          ├── Python
          └── C++17
               ↓
        elementwise compare
               ↓
             PASS
```

## 10. CPU execution model

The naive implementation repeatedly:

```text
loads input
loads weights
multiply-accumulates
writes output
```

Performance depends on cache locality, contiguous layout, vectorization, loop order, data type, threading, and optimized libraries.

## 11. GPU execution model

A GPU path adds system behavior around the same math:

```text
CPU tensor
   ↓
device allocation
   ↓
H2D copy
   ↓
kernel launch
   ↓
parallel GPU computation
   ↓
synchronization when result is needed
   ↓
D2H copy when CPU consumes output
```

GPU work is normally asynchronous. Timing only the launch can measure launch overhead rather than completed computation.

For small workloads, transfer plus launch overhead may exceed compute time. Larger CNNs therefore benefit from keeping intermediate tensors resident on the GPU.

## 12. CPU vs GPU evidence model

Future GPU paths should report:

| Component | Evidence |
|---|---|
| H2D | bytes and time |
| kernel | GPU-event time |
| synchronization | explicit location |
| D2H | bytes and time |
| end-to-end | CPU wall clock |
| warm-up | stated explicitly |
| correctness | compared with CPU oracle |

The course does not treat “CUDA ran” as sufficient evidence.

## 13. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Ruff | Required | Python reference |
| mypy | Required | Typed helpers |
| pytest | Required | Shape/math/PyTorch equivalence |
| CPU PyTorch equivalence | Required | Numerical oracle |
| C++ configure/build/CTest | Required | Native CPU implementation |
| Python ↔ C++ parity | Required | Full tensor comparison |
| CUDA build | Optional | Portable CI must not require GPU |
| GPU correctness | Required when enabled | CPU reference is oracle |
| ONNX | N/A | No learned model artifact required |

## 14. Structure

```text
08_Convolution/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── python/
│   ├── convolution.py
│   ├── convolution_walkthrough.py
│   └── convolution_native_parity.py
├── cpp/
│   └── convolution.cpp
└── tests/
    ├── conftest.py
    └── test_convolution.py
```

## 15. Worked example

Input:

```text
1 2 3
4 5 6
7 8 9
```

Kernel:

```text
1 0
0 -1
```

The output is:

```text
-4 -4
-4 -4
```

## 16. Failure modes

- confusing convolution with cross-correlation
- incorrect output-size math
- CHW vs HWC mismatch
- weight-layout mismatch
- padding or stride off-by-one errors
- forgetting per-output-channel bias
- reading GPU output before synchronization
- measuring only launch latency
- unnecessary H2D/D2H copies
- assuming faster hardware fixes poor memory layout

## 17. Engineering rules

1. write shapes explicitly
2. record layout explicitly
3. prove CPU correctness first
4. compare every element for small deterministic cases
5. keep stride/padding in the interface
6. count memory movement as GPU cost
7. make synchronization explicit
8. distinguish kernel latency from end-to-end latency
9. keep ordinary CI independent of CUDA
10. optimize only after correctness evidence

## 18. Connection to surround perception

```text
surround cameras
      ↓
CNN backbones
      ↓
multi-scale visual features
      ↓
BEV / sensor fusion
      ↓
detection + tracking
```

Convolution shape, layout, memory movement, and native execution directly matter to the later capstone.

## 19. Book takeaway

> **Convolution is simple mathematics repeated at enormous scale. Production performance comes from mapping that repetition onto memory and hardware without changing the numerical contract.**

## 20. Exercises

1. Add dilation.
2. Add grouped convolution.
3. Implement depthwise convolution.
4. Compare CHW and HWC indexing.
5. Benchmark alternate CPU loop orders.
6. Compare float32 and float64.
7. Sketch CUDA block/grid mapping.
8. Estimate transfer cost for a 1920×1080 RGB frame.
9. Explain why intermediate tensors should stay on GPU.
10. Relate depthwise convolution to MobileNet.

## Next

Module 09 builds a complete CNN training loop and connects learned convolution kernels to native inference.
