# Standard Module Template

Use this structure as a target for substantial ML Computer Vision lessons.

```text
XX_Topic/
├── README.md
├── VIDEO.md
├── python/
│   ├── from_scratch.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── cpp/
│   ├── CMakeLists.txt
│   └── inference.cpp
├── tests/
│   ├── test_math.py
│   ├── test_model.py
│   └── test_equivalence.py
├── configs/
│   └── default.yml
└── artifacts/
    └── metadata.example.yml
```

Early conceptual modules do not need every file. The structure grows with the lesson.

## README learning sequence

Each module should follow this narrative:

```text
WHY
 ↓
MATH / INTUITION
 ↓
BUILD THE CORE IDEA
 ↓
FRAMEWORK IMPLEMENTATION
 ↓
TRAIN / EVALUATE
 ↓
EXPORT / PACKAGE
 ↓
NATIVE / GPU VIEW
 ↓
FAILURE MODES
 ↓
EXERCISES
```

## Minimum definition of done

A module is complete when it has:

- explicit learning objectives and prerequisites
- equations/intuition required to understand the method
- runnable code
- deterministic self-tests or unit tests where practical
- expected results or invariant checks
- failure-mode discussion
- reproducible commands
- dependency/data/model provenance
- a VIDEO.md outline
- a native/deployment note for major model modules

## Major model deployment contract

For significant trained models, progressively prove:

```text
PyTorch reference
      ↓
exported model
      ↓
ONNX Runtime Python
      ↓
ONNX Runtime C++
      ↓
GPU / optimized runtime
```

Each transition should have an equivalence tolerance and test evidence rather than assuming export correctness.
