"""Compare Module 35 C++17 output with Python visual-token fusion."""
import math
import subprocess
import sys
from pathlib import Path

from vlm import synthetic


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: vlm_native_parity.py <binary>")
    lines = subprocess.check_output([str(Path(sys.argv[1]))], text=True).splitlines()
    expected = synthetic()
    if len(lines) != len(expected):
        raise AssertionError("token count mismatch")
    for line, token in zip(lines, expected, strict=True):
        actual = [float(value) for value in line.split()]
        if len(actual) != len(token):
            raise AssertionError("embedding width mismatch")
        for a, b in zip(actual, token, strict=True):
            if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError("visual token projection mismatch")
    print("Module 35 Python/C++ parity: PASS")


if __name__ == "__main__":
    main()
