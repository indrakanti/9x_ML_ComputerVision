# Building and Testing

The repository root orchestrates modules; lesson-specific implementation stays inside each module.

## Full portable course gate

```bash
source .venv/bin/activate
bash course.sh test
```

Equivalent manual commands:

```bash
ruff check .
mypy 00_Setup/python 01_TensorsMemory/python 02_ProbabilityStatistics/python
pytest

cmake -S . -B build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
```

## Build one module

Modules with C++ are independently buildable.

Example:

```bash
cmake -S 01_TensorsMemory -B build/module01 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build/module01 --parallel
ctest --test-dir build/module01 --output-on-failure
```

Python-only module:

```bash
pytest 02_ProbabilityStatistics/tests -q
python 02_ProbabilityStatistics/python/statistics_walkthrough.py
```

## Formatting and static analysis

Python:

```bash
ruff check .
mypy 00_Setup/python 01_TensorsMemory/python 02_ProbabilityStatistics/python
```

C++:

```bash
clang-format -i <changed-cpp-files>
cmake -S . -B build-tidy -DCV9X_ENABLE_CLANG_TIDY=ON
cmake --build build-tidy --parallel
```

## CI contract

The portable PR gate follows this sequence:

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

Not every conceptual module has ONNX or C++; those stages are explicitly N/A rather than being satisfied with artificial examples.

GPU correctness/performance remains a separate validation layer and does not make ordinary CPU CI dependent on accelerator availability.
