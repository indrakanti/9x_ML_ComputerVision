"""Verify Module 03 Python optimizer math against the native C++ implementation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from optimization import AdamState, MomentumState, adam_step, gradient_descent_step, momentum_step


def quadratic_grad(value: float) -> float:
    return 2.0 * (value - 3.0)


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

    gd = gradient_descent_step(10.0, quadratic_grad(10.0), 0.1)

    momentum = MomentumState(value=10.0)
    for _ in range(2):
        momentum = momentum_step(
            momentum,
            quadratic_grad(momentum.value),
            learning_rate=0.01,
        )

    adam = AdamState(value=10.0)
    for _ in range(200):
        adam = adam_step(adam, quadratic_grad(adam.value), learning_rate=0.1)

    expected = {
        "gd_one_step": gd,
        "momentum_two_steps": momentum.value,
        "adam_200_steps": adam.value,
    }
    actual = run_native(args.executable)

    for key, expected_value in expected.items():
        actual_value = float(actual[key])
        if not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{key}: C++={actual_value} Python={expected_value}")

    print("Module 03 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
