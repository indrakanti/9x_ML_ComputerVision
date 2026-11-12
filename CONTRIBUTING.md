# Contributing to 9x ML Computer Vision

This repository is an engineering-oriented ML Computer Vision course. Contributions should preserve four properties: **learnability, mathematical clarity, reproducibility, and deployability**.

## Baseline

- Linux
- Python 3.10+
- PyTorch
- C++17 baseline
- CMake 3.20+
- CPU correctness path required
- GPU-specific paths optional unless a lesson explicitly targets acceleration

Run the complete portable gate:

```bash
./course.sh doctor
./course.sh test
```

## Module ownership

Anything specific to a lesson belongs inside that module: Python, C++, tests, configs, metadata, examples and lesson artifacts. Root-level code is reserved for orchestration or infrastructure that is genuinely shared across modules.

## Standard quality gate

Each module PR is checked against:

```text
Python formatting / lint
mypy
pytest
CPU PyTorch tests
ONNX tests
C++ configure
C++ build
CTest
Python ↔ exported-model equivalence
```

A gate may be marked N/A only when the module genuinely has no relevant model/export/native component.

## Standard lesson contract

Every major lesson should answer:

1. Why does this technique exist?
2. What is the mathematical model?
3. Can we build the core idea from first principles?
4. How does PyTorch express the same idea?
5. How is it trained and evaluated?
6. What fails, and why?
7. How is the model exported or packaged?
8. What changes in C++ / GPU / production inference?

Use [MODULE_TEMPLATE.md](MODULE_TEMPLATE.md) when adding a module.

## Pull requests

Keep PRs focused. Changes should:

- include deterministic tests where practical
- preserve a CPU smoke/correctness path
- avoid requiring a GPU for ordinary CI
- document new dependencies
- record dataset/model provenance
- keep generated checkpoints and large artifacts out of Git
- update course documentation when interfaces or commands change

## External datasets, models and media

The repository Apache-2.0 license applies to repository-owned source code and documentation unless a file states otherwise.

It does **not** automatically license third-party datasets, pretrained weights, images, videos, annotations, papers, figures, logos, fonts, or model artifacts.

For every external dependency, document:

- source URL
- version/revision
- license
- attribution requirements
- redistribution permission
- checksum when practical

Prefer reproducible download/preparation scripts over committing third-party assets.

## Reproducibility

Normal dependency files specify supported ranges. Release-quality course checkpoints should additionally record a known-good environment snapshot under `locks/`.

A result is not reproducible unless the relevant dataset version, preprocessing, split, seed, model configuration, software environment, device, precision and metrics are known.

## Contribution licensing

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion are provided under Apache License 2.0, consistent with Section 5 of the license.
