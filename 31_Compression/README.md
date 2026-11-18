# Module 31 — Distillation, Pruning, and Quantization

Stage 1 closes by asking the deployment question that follows training and adaptation:

> How do we make a model smaller or cheaper without losing the behavior we care about?

This module covers three complementary techniques:

```text
teacher → student distillation
weights → pruning mask
float weights → INT8 quantization
```

## Learning objectives

You should be able to:

- explain knowledge distillation
- compute temperature-scaled teacher/student distributions
- understand soft-target cross entropy and the $T^2$ scaling convention
- apply magnitude pruning
- compute sparsity
- perform symmetric per-tensor INT8 quantization
- dequantize and measure reconstruction error
- reproduce all three transforms in C++17
- distinguish compression metrics from end-to-end deployment benefit

## 1. Knowledge distillation

A student can learn from a teacher's soft class distribution.

For logits $z_t$ and $z_s$ at temperature $T$:

$$
p_t=\operatorname{softmax}(z_t/T)
$$

$$
p_s=\operatorname{softmax}(z_s/T)
$$

A common distillation term is:

$$
L_{KD}
=
T^2
\left(
-\sum_i p_t(i)\log p_s(i)
\right)
$$

The $T^2$ factor compensates for gradient scaling introduced by temperature.

Real training often combines distillation with the ordinary supervised loss.

## 2. Why soft targets help

The teacher distribution contains information beyond the winning class.

For example:

```text
car        0.80
truck      0.15
pedestrian 0.05
```

communicates class similarity that a one-hot label does not.

## 3. Magnitude pruning

Unstructured magnitude pruning removes small weights:

$$
w'=
\begin{cases}
0,&|w|<\tau\\
w,&otherwise
\end{cases}
$$

Sparsity is:

$$
s=\frac{N_{zero}}{N}
$$

High sparsity does **not** guarantee lower latency. The runtime/hardware must exploit sparse structure.

## 4. Structured vs unstructured pruning

Unstructured:
- removes individual weights
- easy mathematically
- may require specialized sparse kernels

Structured:
- removes channels, blocks, heads, or filters
- changes tensor dimensions
- often easier for dense hardware to accelerate

This executable example uses unstructured pruning because the semantics are transparent.

## 5. Symmetric INT8 quantization

For floating-point values $x$:

$$
s=\frac{\max|x|}{127}
$$

$$
q=\operatorname{clip}
\left(
\operatorname{round}(x/s),
-127,
127
\right)
$$

Dequantization:

$$
\hat{x}=sq
$$

This module uses symmetric zero-point-free quantization.

## 6. Quantization error

Compression is not free.

The module reports maximum absolute dequantization error:

$$
E_{max}=\max_i|x_i-\hat{x}_i|
$$

Real models also require task-level accuracy evaluation after quantization.

## 7. Calibration and QAT

Post-training quantization (PTQ) derives ranges from calibration data.

Quantization-aware training (QAT) simulates quantization during training so the model can adapt.

This module demonstrates the arithmetic contract; later deployment modules cover runtime-specific quantized execution.

## 8. Executable reference

The artifact contains:

- teacher logits
- student logits
- distillation temperature
- model weights
- pruning threshold
- quantization weights

Python computes:

- teacher/student probabilities
- distillation loss
- pruned weights
- sparsity
- INT8 values
- scale
- dequantized values
- maximum quantization error

PyTorch verifies the distillation loss.

C++17 reproduces the compression math.

## 9. Deployment implications

Compression should be evaluated as a complete system:

```text
model size
latency
memory
power
accuracy
runtime support
determinism
```

A smaller file is not necessarily a faster model.

## 10. Failure modes

- distilling from probabilities instead of teacher logits incorrectly
- forgetting temperature scaling
- pruning without fine-tuning/recovery evaluation
- counting zeros as a speedup without sparse kernels
- mixing per-tensor and per-channel quantization scales
- using different rounding conventions across runtimes
- quantizing unsupported operators and silently falling back to float
- measuring model size but not end-to-end latency

## 11. Engineering rules

1. version distillation temperature
2. preserve teacher/student class ordering
3. version pruning threshold and structure
4. record achieved sparsity
5. record quantization scheme and scale granularity
6. compare dequantized numerics and task metrics
7. measure speedup on the actual target runtime

## 12. AV relevance

Distillation and quantization are especially important when the same perception stack must fit within embedded compute, thermal, and power budgets. Compression decisions should be tied to system-level latency and safety evidence, not just parameter count.

## 13. Book takeaway

> **Compression is a contract between learning quality and runtime capability. Distillation changes supervision, pruning changes structure, and quantization changes number representation—each requires independent validation.**

## Reproducible commands

```bash
python 31_Compression/python/compression_walkthrough.py
pytest 31_Compression/tests
cmake -S 31_Compression -B build-m31 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m31
ctest --test-dir build-m31 --output-on-failure
python 31_Compression/python/compression_native_parity.py \
  build-m31/cv9x_compression_math \
  31_Compression/artifacts/compression.txt
```

## Stage 1 complete

Modules 00–31 now cover the path from tensors and optimization through modern backbones, detection/segmentation, self-supervision, PEFT, and compression.
