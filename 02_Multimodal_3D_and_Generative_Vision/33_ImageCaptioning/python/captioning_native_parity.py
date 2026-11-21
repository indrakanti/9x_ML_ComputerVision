"""Check Module 33 Python/C++ attention fixture agreement."""
import math
import subprocess
import sys
from pathlib import Path

from captioning import synthetic


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: captioning_native_parity.py <binary>")
    output = subprocess.check_output([str(Path(sys.argv[1]))], text=True)
    actual = [float(line) for line in output.splitlines()]
    expected = synthetic()
    if len(actual) != len(expected):
        raise AssertionError("native fixture length mismatch")
    for a, b in zip(actual, expected, strict=True):
        if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError(f"captioning parity mismatch: {a} != {b}")
    print("Module 33 Python/C++ attention parity: PASS")


if __name__ == "__main__":
    main()
