"""Verify Module 12 Python residual block against C++17."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from residual_learning import example_input, load_artifact, manual_residual_forward


def run_native(executable: Path, artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    expected = manual_residual_forward(artifact, example_input())
    actual = run_native(args.executable, args.artifact)["values"]

    if len(actual) != len(expected):
        raise AssertionError("output size mismatch")
    for index, (cpp_value, python_value) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if not math.isclose(
            float(cpp_value),
            python_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"output[{index}] C++={cpp_value} Python={python_value}"
            )
    print("Module 12 Python ↔ C++ residual parity: PASS")


if __name__ == "__main__":
    main()
