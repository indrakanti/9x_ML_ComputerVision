# Native C++ / GPU Runtime Track

The ML course is not Python-only.

Python/PyTorch is the preferred training and experimentation environment, while the production runtime track is C++.

## Native progression

~~~text
C++17/20 + CMake
    |
tensor / image memory contracts
    |
LibTorch concepts and/or exported-model runtime
    |
ONNX Runtime C++
    |
CUDA execution providers
    |
TensorRT C++
    |
streaming perception process
~~~

## Why C++ is present from Module 00

Safety-oriented and real-time systems care about more than model accuracy:

- bounded interfaces
- explicit ownership
- predictable lifecycle
- thread/scheduling control
- allocator behavior
- accelerator synchronization
- fault handling
- deterministic startup/shutdown
- integration with platform middleware
- diagnosability and evidence

C++ is therefore a first-class runtime language throughout the course.

## Build the native sanity target

~~~bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
~~~

Direct run:

~~~bash
./build-native/cv9x_native_sanity --self-test
~~~

## GPU policy

GPU is also first-class, but the repository separates:

~~~text
CPU correctness
GPU correctness
GPU performance
~~~

CPU correctness stays in portable CI.

CUDA/TensorRT validation requires a compatible NVIDIA environment and is introduced through dedicated GPU checks rather than making every contributor depend on a GPU.

## CUDA toolchain probe

When the NVIDIA CUDA toolkit and nvcc are installed:

~~~bash
cmake -S . -B build-cuda   -DCMAKE_BUILD_TYPE=Release   -DCV9X_ENABLE_CUDA=ON
~~~

The configure step fails explicitly if CUDA is requested but no CUDA compiler is available.

Later course PRs will add real CUDA/ONNX Runtime/TensorRT targets behind the same opt-in mechanism.
