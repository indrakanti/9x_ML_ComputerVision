"""Verify Module 02 Python statistics against the native C++ implementation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from statistics_helpers import accuracy_confidence_interval, correlation, covariance


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

    x = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    ci_low, ci_high = accuracy_confidence_interval(correct=90, total=100)

    expected = {
        "mean_x": float(x.mean()),
        "variance_x": float(x.var()),
        "covariance_xy": covariance(x, y),
        "correlation_xy": correlation(x, y),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    actual = run_native(args.executable)

    for key, expected_value in expected.items():
        actual_value = float(actual[key])
        if not np.isclose(actual_value, expected_value, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"{key}: C++={actual_value} Python={expected_value}")

    print("Module 02 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
