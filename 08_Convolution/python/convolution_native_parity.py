"""Verify Module 08 Python convolution against C++17."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from convolution import conv2d_chw, example_tensors


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

    image, weights, bias, config = example_tensors()
    expected = conv2d_chw(image, weights, bias, config)
    actual = run_native(args.executable)

    if actual["shape"] != list(expected.shape):
        raise AssertionError(
            f"shape mismatch: C++={actual['shape']} Python={list(expected.shape)}"
        )

    flat_expected = expected.reshape(-1).tolist()
    flat_actual = actual["values"]
    if len(flat_actual) != len(flat_expected):
        raise AssertionError("flattened output size mismatch")

    for index, (cpp_value, python_value) in enumerate(
        zip(flat_actual, flat_expected, strict=True)
    ):
        if not math.isclose(
            float(cpp_value),
            float(python_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"output[{index}]: C++={cpp_value} Python={python_value}"
            )

    print("Module 08 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
