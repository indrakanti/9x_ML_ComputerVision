# Runtime Architecture — Training, Native Inference, GPU and Safety-Oriented Integration

## Core architecture

The course uses different tools for different responsibilities.

~~~text
                  OFFLINE / DEVELOPMENT

 dataset
   |
 Python + PyTorch
   |
 training / validation / calibration
   |
 checkpoint
   |
 export + compatibility tests
   |
 ONNX / deployment artifact
   |
   +--------------------------------------------------+
                                                      |
                                      ONLINE / PRODUCTION
                                                      |
                                                C++ runtime
                                                      |
                                      preprocessing / tensors
                                                      |
                                    ONNX Runtime / TensorRT
                                                      |
                                                  GPU
                                                      |
                                      postprocessing / outputs
                                                      |
                              monitoring / timing / plausibility
                                                      |
                                       downstream application
~~~

## Python responsibility

Python is optimized for:

- model development
- training
- experimentation
- visualization
- dataset tooling
- metric analysis
- checkpoint creation
- export
- offline validation

Python is not assumed to be the final production runtime.

## C++ responsibility

C++ is optimized for the production path:

- process lifecycle
- model loading/version checks
- image/tensor memory contracts
- preprocessing/postprocessing
- native inference APIs
- thread ownership
- queueing/backpressure
- accelerator interaction
- timestamps and freshness
- runtime monitoring
- diagnostics
- integration with platform middleware

## GPU responsibility

The GPU executes computationally intensive tensor/model workloads.

The course will explicitly teach that GPU use introduces systems behavior that must be understood:

- host-to-device and device-to-host transfer
- device memory allocation
- pinned host memory
- asynchronous copies
- streams
- events
- synchronization
- kernel launch overhead
- warm-up
- context creation
- precision modes
- batching
- workspace/scratch memory
- engine/context lifetime
- contention
- error propagation

## Model artifact contract

A model file alone is insufficient.

A deployable artifact contract should include:

~~~text
model format/version
input names
input shapes
input dtype
layout: NCHW/NHWC
color order
normalization
resize/crop policy
output names
output shapes
label mapping
postprocessing version
precision
runtime compatibility
calibration data/version when quantized
training provenance
validation evidence
~~~

Python export and C++ import tests should verify this contract.

## Safety-oriented distinction

Neither C++ nor GPU execution automatically makes an ML component suitable for a safety-critical system.

The system architecture must separate:

~~~text
model output
from
evidence that the output is timely, fresh, plausible and usable
~~~

Production lessons will therefore introduce checks such as:

- input freshness
- timestamp monotonicity
- model-execution timeout
- output finite/range checks
- tensor shape/type checks
- confidence/quality checks
- geometry/plausibility checks
- resource monitoring
- GPU/runtime error detection
- model/config compatibility
- watchdog/liveness
- degraded behavior / fallback
- fault reporting

## Verification layers

The course will keep these evidence classes separate.

### 1. Algorithm correctness

Does the model/algorithm produce the expected result?

### 2. Export equivalence

Does the exported model agree with the training-framework reference within tolerance?

### 3. Native runtime correctness

Does C++ preprocessing/inference/postprocessing match the Python reference?

### 4. GPU correctness

Does accelerated execution preserve required numerical behavior?

### 5. Performance evidence

Measure:

- mean latency
- median
- p95
- p99
- maximum observed
- throughput
- memory
- warm-up
- copy time
- compute time
- synchronization time

### 6. System integration evidence

Check:

- deadlines
- queue bounds
- dropped/stale data
- lifecycle
- restart/recovery
- monitoring
- fault containment

## Course implementation pattern

Major model lessons should evolve toward a structure like:

~~~text
XX_Topic/
├── README.md
├── VIDEO.md
├── python/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── inference.cpp
│   └── preprocessing.cpp
├── tests/
│   ├── test_python.py
│   └── test_equivalence.py
└── artifacts/
    └── metadata.example.yml
~~~

Generated models/checkpoints remain outside source control unless a deliberately small test artifact is required.

## Deployment progression

The native track will grow in stages:

~~~text
C++ standard-library sanity
-> image/tensor contracts
-> native tensor/runtime concepts
-> ONNX export
-> ONNX Runtime C++ CPU
-> ONNX Runtime CUDA
-> TensorRT
-> mixed precision / INT8
-> streaming inference
-> profiling
-> monitoring / degraded behavior
-> edge/embedded integration
~~~

This lets students understand both the model and the system that has to run it.
