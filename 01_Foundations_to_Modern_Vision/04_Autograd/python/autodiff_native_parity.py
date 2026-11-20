"""Verify Module 04 Python autodiff against the native C++ implementation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from autodiff import Value


def run_native(executable: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()

    x = Value(2.0)
    y = Value(3.0)
    loss = (x * y + x) ** 2
    loss.backward()

    expected = {"loss": loss.data, "grad_x": x.grad, "grad_y": y.grad}
    actual = run_native(args.executable)

    for key, expected_value in expected.items():
        actual_value = float(actual[key])
        if not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{key}: C++={actual_value} Python={expected_value}")

    print("Module 04 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
