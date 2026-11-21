"""Cross-language parity of open-vocabulary detection fixture."""
import math
import subprocess
import sys
from pathlib import Path

from open_vocab import synthetic


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: open_vocab_native_parity.py <binary>")
    lines = subprocess.check_output([str(Path(sys.argv[1]))], text=True).splitlines()
    expected = synthetic()
    if len(lines) != len(expected):
        raise AssertionError("detection count mismatch")
    for line, (index, label, score) in zip(lines, expected, strict=True):
        parts = line.split()
        if len(parts) != 3 or int(parts[0]) != index or int(parts[1]) != label:
            raise AssertionError("detection identity mismatch")
        if not math.isclose(float(parts[2]), score, rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError("detection score mismatch")
    print("Module 36 Python/C++ detection parity: PASS")


if __name__ == "__main__":
    main()
