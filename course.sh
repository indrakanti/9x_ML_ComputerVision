#!/usr/bin/env bash
set -euo pipefail

COMMAND=${1:-help}

activate_venv() {
  if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv is missing. Run ./setup.sh first."
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

case "$COMMAND" in
  doctor)
    activate_venv
    python 00_Setup/python/environment_check.py
    echo
    echo "Native toolchain:"
    cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
    cmake --build build-native --parallel
    ctest --test-dir build-native --output-on-failure
    echo
    echo "COURSE ENVIRONMENT READY"
    ;;
  test)
    activate_venv
    ruff check .
    mypy 00_Setup/python 01_TensorsMemory/python 02_ProbabilityStatistics/python
    pytest
    cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
    cmake --build build-native --parallel
    ctest --test-dir build-native --output-on-failure
    ;;
  clean)
    rm -rf build build-native build-native-werror build-cuda .pytest_cache .mypy_cache .ruff_cache htmlcov
    ;;
  help|*)
    cat <<'EOF'
9x ML Computer Vision course helper

  ./course.sh doctor   verify Python + ML + native C++ environment
  ./course.sh test     run lint, type checks, Python tests, C++ build and CTest
  ./course.sh clean    remove local generated build/test outputs

Setup:
  ./setup.sh --cpu
  ./setup.sh --nvidia
EOF
    ;;
esac
