# Module 01 — Images, Tensors, Memory Layout and Device Semantics

This module builds the memory model used by every later ML Computer Vision lesson.

## Learning objectives

By the end of this module you should be able to:

- explain the difference between an image and a tensor
- read tensor shapes without guessing
- distinguish HWC, CHW and NCHW layouts
- compute and interpret element strides
- explain contiguous vs non-contiguous tensors
- distinguish a view from a copy
- convert uint8 image data to float32 model input correctly
- explain CPU and GPU device placement
- describe a tensor to C++ using shape, stride, dtype and ownership
- avoid common preprocessing mismatches between Python and native inference

## 1. Why this matters

A model does not receive "an image" in the abstract. It receives memory.

A useful tensor contract is:

~~~text
data pointer / storage
shape
strides
dtype
layout
device
ownership / lifetime
normalization
~~~

Two arrays can have the same visible values and the same shape while having different strides or ownership semantics.

That becomes important when we cross boundaries:

~~~text
OpenCV / NumPy
      ↓
PyTorch
      ↓
ONNX
      ↓
C++ runtime
      ↓
GPU
~~~

## 2. Image layout

An RGB image is commonly stored in NumPy/OpenCV-like form as:

~~~text
H × W × C
~~~

For example:

~~~text
480 × 640 × 3
~~~

PyTorch vision models usually use:

~~~text
N × C × H × W
~~~

A single image therefore typically moves through:

~~~text
HWC uint8
   ↓ convert dtype
HWC float32
   ↓ permute
CHW float32
   ↓ add batch dimension
NCHW float32
~~~

## 3. Strides

For a contiguous tensor, a stride tells us how many **elements** we move in storage when an index changes by one.

For shape:

~~~text
(2, 3, 4)
~~~

a contiguous row-major element stride is:

~~~text
(12, 4, 1)
~~~

The address of element `(n, c, h)` relative to the base is:

[
	ext{offset}=n s_0 + c s_1 + h s_2
]

where (s_i) is the stride for dimension (i).

## 4. Views, copies and contiguity

NumPy and PyTorch can often change the logical interpretation of memory without copying it.

Examples:

- slicing may produce a view
- `torch.from_numpy(...)` can share CPU memory
- `permute(...)` changes strides and is usually non-contiguous
- `contiguous()` materializes a contiguous copy when required

This matters for native runtimes because a pointer alone is not enough to interpret a non-contiguous tensor.

## 5. Dtype conversion and normalization

Camera/image data commonly starts as `uint8` in the range ([0,255]).

A common model-input conversion is:

[
x_f = rac{x_{u8}}{255}
]

followed by channel normalization:

[
x'_c = rac{x_c-mu_c}{sigma_c}
]

The exact order, color channel convention, resize/crop policy and constants are part of the **model contract**.

## 6. CPU and GPU device semantics

PyTorch tensors live on a device.

~~~python
x = x.to("cuda")
~~~

is not merely a label change. Data must become available in device memory.

Keep these concepts separate:

~~~text
logical tensor
storage
device
copy / transfer
asynchronous execution
synchronization
~~~

This module uses CPU as the correctness baseline. GPU-specific performance comes later.

## 7. Run the demonstrations

~~~bash
source .venv/bin/activate

python 01_TensorsMemory/python/tensor_walkthrough.py
pytest 01_TensorsMemory/tests -q
~~~

Native C++:

~~~bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
./build-native/cv9x_tensor_contract --self-test
ctest --test-dir build-native --output-on-failure
~~~

## 8. What the walkthrough demonstrates

The Python walkthrough shows:

- HWC NumPy image creation
- shared-memory conversion to PyTorch
- mutation visibility across NumPy/PyTorch
- HWC → CHW using `permute`
- non-contiguous stride behavior
- `contiguous()`
- uint8 → float32 conversion
- NCHW batching
- CPU/GPU device discovery
- a serializable tensor contract

## 9. C++ tensor contract

The native example intentionally does **not** depend on LibTorch yet.

It introduces the runtime metadata we will reuse later:

~~~text
TensorContract
├── shape
├── strides
├── dtype
├── layout
├── device
└── owns_memory
~~~

Later ONNX Runtime/TensorRT modules will bind this contract to real runtime tensors.

## 10. Common failure modes

### HWC passed to a model expecting NCHW

The values are valid, but the interpretation is wrong.

### RGB/BGR mismatch

OpenCV commonly exposes BGR while many training pipelines use RGB.

### uint8 arithmetic

Dividing or normalizing before converting to floating point can produce the wrong result depending on the operation.

### Non-contiguous input

Some kernels/runtimes require contiguous storage. `permute` changes logical layout without guaranteeing contiguous storage.

### Lifetime bug

A tensor/view can reference memory owned elsewhere. Native code must not outlive the underlying storage.

### Silent preprocessing drift

Python and C++ may both "work" while using different resize, channel order, normalization or dtype rules.

## 11. Engineering invariant

From this module onward, never describe model input only as:

> 224 × 224 image

Describe it as a contract, for example:

~~~text
shape: [1, 3, 224, 224]
layout: NCHW
dtype: float32
device: CPU/CUDA
color: RGB
range: [0, 1]
normalization:
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
~~~

## 12. Exercises

1. Create an HWC image and manually calculate its contiguous strides.
2. Permute it to CHW and explain why the tensor becomes non-contiguous.
3. Verify that `contiguous()` preserves values but changes storage/strides.
4. Demonstrate shared storage with `torch.from_numpy`.
5. Write a preprocessing function that returns NCHW float32.
6. Extend the C++ contract with byte-size validation.
7. Explain why shape equality is insufficient for native-runtime equivalence.

## Next

Module 02 introduces probability and statistics for ML vision: distributions, expectation, variance, conditional probability, likelihood and the statistics behind training/evaluation.
