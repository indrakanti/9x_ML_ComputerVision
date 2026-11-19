# Episode 01 — Images to PyTorch Tensors: Shape, Stride, Dtype and Memory

Suggested title:

**Images Are Memory: NumPy → PyTorch Tensors → C++**

Suggested thumbnail:

**SHAPE IS NOT ENOUGH**

Target duration: 30–38 minutes.

## Chapters

~~~text
00:00 Why tensor memory matters
03:00 HWC images
06:00 NumPy shape and byte strides
10:00 torch.from_numpy and shared storage
14:00 CHW / NCHW
18:00 permute, views and non-contiguous tensors
23:00 contiguous copies
26:00 uint8 → float32 → normalization
30:00 CPU vs GPU device semantics
33:00 C++ tensor contract
36:00 Failure modes and Module 02
~~~

## Demonstration

~~~bash
python 01_TensorsMemory/python/tensor_walkthrough.py
pytest 01_TensorsMemory/tests -q

cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
./build-native/cv9x_tensor_contract --self-test
~~~

## Core message

A tensor is not just numbers and shape.

For production ML vision, always reason about:

~~~text
shape
stride
dtype
layout
device
ownership
preprocessing contract
~~~
