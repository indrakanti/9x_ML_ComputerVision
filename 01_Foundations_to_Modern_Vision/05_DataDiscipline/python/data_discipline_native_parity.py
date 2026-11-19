"""Verify Module 05 Python behavior against the native C++ implementation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from data_discipline import assign_split, binary_metrics


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

    sample_ids = ["frame_0001", "frame_0002", "frame_0003", "frame_0004", "frame_0005"]
    metrics = binary_metrics(
        truth=[1, 1, 1, 0, 0, 0],
        prediction=[1, 1, 0, 1, 0, 0],
    )

    expected_splits = {sample_id: assign_split(sample_id) for sample_id in sample_ids}
    expected_metrics = {
        "tp": metrics.true_positive,
        "tn": metrics.true_negative,
        "fp": metrics.false_positive,
        "fn": metrics.false_negative,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "specificity": metrics.specificity,
        "f1": metrics.f1,
    }

    actual = run_native(args.executable)

    if actual["splits"] != expected_splits:
        raise AssertionError(f"split mismatch: C++={actual['splits']} Python={expected_splits}")

    actual_metrics = actual["metrics"]
    for key, expected_value in expected_metrics.items():
        actual_value = actual_metrics[key]
        if isinstance(expected_value, int):
            if int(actual_value) != expected_value:
                raise AssertionError(f"{key}: C++={actual_value} Python={expected_value}")
        elif not math.isclose(float(actual_value), expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{key}: C++={actual_value} Python={expected_value}")

    print("Module 05 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
