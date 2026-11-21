"""Compare Python embedding scores with the native C++17 fixture."""
import math
import subprocess
import sys
from pathlib import Path

from zero_shot import scores


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: zero_shot_native_parity.py <native executable>")
    binary = Path(sys.argv[1])
    output = subprocess.check_output([str(binary)], text=True)
    actual = [float(line) for line in output.splitlines()]
    expected = scores([3, 4], [[3, 4], [4, -3], [-3, -4]])
    if len(actual) != len(expected):
        raise AssertionError("native score count mismatch")
    for a, b in zip(actual, expected, strict=True):
        if not math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6):
            raise AssertionError(f"native/Python mismatch: {a} vs {b}")
    print("Module 32 Python/C++ parity: PASS")


if __name__ == "__main__":
    main()
