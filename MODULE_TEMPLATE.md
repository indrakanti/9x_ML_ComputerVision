# Standard Module Template

Use this structure as a target for substantial ML Computer Vision lessons.

```text
XX_Topic/
├── README.md
├── VIDEO.md
├── CMakeLists.txt            # when native C++ exists
├── python/
│   ├── from_scratch.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── cpp/
│   └── inference.cpp
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

Early conceptual modules do not need every file. The structure grows with the lesson.

## README formatting standard

Use GitHub-flavored Markdown that renders cleanly in the repository:

- use `$ ... $` for display equations
- use `$...# Standard Module Template

Use this structure as a target for substantial ML Computer Vision lessons.

```text
XX_Topic/
├── README.md
├── VIDEO.md
├── CMakeLists.txt            # when native C++ exists
├── python/
│   ├── from_scratch.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── cpp/
│   └── inference.cpp
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

Early conceptual modules do not need every file. The structure grows with the lesson.

 for inline math
- use fenced code blocks with a language such as `bash`, `python`, `cpp`, or `text`
- keep headings sequential and descriptive
- use tables for runtime applicability / gate status
- avoid raw bracketed pseudo-LaTeX such as `[ equation ]`
- preview the README on GitHub before merging

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


## Module ownership rule

Anything that exists specifically to teach one module belongs inside that module: Python, C++, tests, configs, metadata and lesson artifacts. Root-level code is reserved for orchestration or genuinely shared infrastructure.

## Standard module quality gate

Every module PR is evaluated against:

~~~text
Python formatting / lint
mypy
pytest
CPU PyTorch tests
ONNX tests
C++ configure
C++ build
CTest
Python ↔ exported-model equivalence
~~~

A gate may be explicitly **N/A** when the module genuinely has no model/export/native component. Do not create artificial ONNX or C++ work merely to satisfy a checkbox.
