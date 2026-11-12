# Standard Module Template

Use this structure as the default for substantial ML Computer Vision lessons.

```text
XX_Topic/
├── README.md
├── VIDEO.md
├── CMakeLists.txt
├── python/
│   ├── from_scratch.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── <topic>_native_parity.py
├── cpp/
│   └── native_counterpart.cpp
├── tests/
│   ├── conftest.py
│   ├── test_math.py
│   ├── test_model.py
│   ├── test_onnx.py
│   └── test_equivalence.py
├── configs/
│   └── default.yml
└── artifacts/
    └── metadata.example.yml
```

Not every module needs every file, but **native C++ is the default companion track whenever the concept can be represented meaningfully in native code**.

## Course implementation principle

The course has three synchronized engineering tracks:

```text
Python / PyTorch
learning + training reference
        ↓
C++17 native counterpart
engineering + deployment semantics
        ↓
GPU / optimized runtime
acceleration + production performance
```

ONNX is introduced only when there is a real model artifact to export. C++ is **not** dependent on ONNX.

## README formatting standard

Use GitHub-flavored Markdown that renders cleanly:

- use `$$ ... $$` for display equations
- use `$...$` for inline math
- use fenced code blocks with a language tag
- keep headings sequential
- use tables for runtime applicability and quality gates
- avoid pseudo-LaTeX such as `[ equation ]`
- preview README rendering before merge

## Book-quality README standard

The README is both the runnable lesson and the first draft of the corresponding book chapter. A substantial module README should therefore contain:

- a strong opening motivation: why the topic matters
- explicit learning objectives
- visual/intuitive explanation before implementation details
- equations rendered with GitHub-supported math
- at least one worked numerical or engineering example
- clear Python/reference implementation discussion
- clear C++17/native counterpart discussion
- Python ↔ C++ parity explanation where applicable
- deployment implications, not only training implications
- failure modes and misleading-result examples
- engineering rules that carry forward into later modules
- reproducible commands
- exercises that extend both theory and implementation
- a short **Book takeaway** summarizing the durable principle

Prefer explanations that can be reused in a technical book with minimal rewriting. Avoid README text that is only a directory listing or API reference.

## README learning sequence

```text
WHY
 ↓
MATH / INTUITION
 ↓
PYTHON FROM-SCRATCH IMPLEMENTATION
 ↓
PYTORCH / FRAMEWORK IMPLEMENTATION
 ↓
C++ NATIVE COUNTERPART
 ↓
PYTHON ↔ C++ PARITY
 ↓
TRAIN / EVALUATE
 ↓
EXPORT / PACKAGE        # when a model artifact exists
 ↓
ONNX / C++ RUNTIME      # when export is applicable
 ↓
GPU / OPTIMIZED RUNTIME # when acceleration is applicable
 ↓
FAILURE MODES
 ↓
EXERCISES
```

## Minimum definition of done

A substantial module is complete when it has:

- explicit learning objectives and prerequisites
- equations and intuition
- runnable Python code
- deterministic Python tests
- meaningful C++17 counterpart when technically applicable
- standalone module CMake build
- root CMake integration
- native self-test through CTest
- Python ↔ C++ numerical parity where practical
- CPU PyTorch equivalence when PyTorch is involved
- failure-mode discussion
- reproducible commands
- dependency/data/model provenance
- a VIDEO.md outline

## Native counterpart rule

Do **not** omit C++ merely because a lesson is primarily mathematical or training-oriented.

Examples:

- statistics → native metric/statistics accumulator
- optimization → native optimizer state/update math
- autograd → native reverse-mode graph
- convolution → native CPU/GPU kernel behavior
- metrics → native evaluation/telemetry
- model inference → ONNX Runtime/TensorRT C++ deployment

C++ may be marked N/A only when a native implementation would be genuinely meaningless. The PR description must explain why.

## Model deployment contract

For trained models, progressively prove:

```text
PyTorch reference
      ↓
exported model
      ↓
ONNX Runtime Python
      ↓
ONNX Runtime C++
      ↓
GPU / TensorRT
```

Every transition requires numerical equivalence evidence and explicit tolerance.

## Module ownership rule

Anything specific to one lesson belongs inside that module: Python, C++, tests, CMake, configs, parity tools, metadata and artifacts. Root-level code is only orchestration or genuinely shared infrastructure.

## Standard quality gate

```text
Python formatting / lint        required
mypy                           required
pytest                         required
CPU PyTorch tests              when PyTorch is used
C++ configure                  required when native counterpart applies
C++ build                      required when native counterpart applies
CTest                          required when native counterpart applies
Python ↔ C++ numerical parity  required where practical
ONNX tests                     when a model is exported
Python ↔ ONNX equivalence      when a model is exported
ONNX Python ↔ C++ equivalence  when native ONNX runtime is introduced
GPU correctness                when GPU path is introduced
performance evidence           when performance is a lesson objective
```
