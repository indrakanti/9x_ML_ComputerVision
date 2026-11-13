#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-preflight}"

echo "==> Ruff"
ruff check .

echo "==> mypy"
mypy [0-9][0-9]_*/python

echo "==> pytest"
pytest

echo "==> configure native C++"
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON

echo "==> build native C++"
cmake --build "${BUILD_DIR}" --parallel

echo "==> Python ↔ C++ parity"
python 02_ProbabilityStatistics/python/statistics_native_parity.py \
  "${BUILD_DIR}/02_ProbabilityStatistics/cv9x_statistics"
python 03_Optimization/python/optimization_native_parity.py \
  "${BUILD_DIR}/03_Optimization/cv9x_optimizer_state"
python 04_Autograd/python/autodiff_native_parity.py \
  "${BUILD_DIR}/04_Autograd/cv9x_autodiff"
python 05_DataDiscipline/python/data_discipline_native_parity.py \
  "${BUILD_DIR}/05_DataDiscipline/cv9x_data_discipline"
python 06_LinearClassifiers/python/linear_classifier_native_parity.py \
  "${BUILD_DIR}/06_LinearClassifiers/cv9x_linear_classifier" \
  06_LinearClassifiers/artifacts/linear_binary_model.txt

echo "==> CTest"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo
echo "PRE-PR PREFLIGHT: PASS"
