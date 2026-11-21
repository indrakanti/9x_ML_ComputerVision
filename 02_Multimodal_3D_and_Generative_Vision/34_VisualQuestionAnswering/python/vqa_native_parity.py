"""Verify C++17/Python Module 34 deterministic fixture parity."""
import math
import subprocess
import sys
from pathlib import Path

from vqa import synthetic


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: vqa_native_parity.py <binary>")
    lines = subprocess.check_output([str(Path(sys.argv[1]))], text=True).splitlines()
    answer, weights = synthetic()
    if len(lines) != len(weights) + 1 or lines[0] != answer:
        raise AssertionError("answer or output count differs")
    for actual, expected in zip(lines[1:], weights, strict=True):
        if not math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError("Python/C++ attention differs")
    print("Module 34 Python/C++ VQA parity: PASS")


if __name__ == "__main__":
    main()
