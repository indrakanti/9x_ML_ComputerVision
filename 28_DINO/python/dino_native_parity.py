"""Module 28 Python ↔ C++ DINO parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from dino_self_supervised import (
    ema_teacher_params,
    load_artifact,
    student_distribution,
    symmetric_cross_view_loss,
    teacher_distribution,
)


def run_native(executable: Path, artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _compare_vector(
    actual: list[Any],
    expected: list[float],
    name: str,
) -> None:
    for cpp, py in zip(actual, expected, strict=True):
        if not math.isclose(
            float(cpp),
            py,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{name} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    expected_student = student_distribution(
        artifact.student_view_a,
        artifact.student_temperature,
    )
    expected_teacher = teacher_distribution(
        artifact.teacher_view_b,
        artifact.center,
        artifact.teacher_temperature,
    )
    expected_loss = symmetric_cross_view_loss(artifact)
    expected_ema = ema_teacher_params(artifact)
    actual = run_native(args.executable, args.artifact)

    _compare_vector(actual["student_a"], expected_student, "student")
    _compare_vector(actual["teacher_b"], expected_teacher, "teacher")
    _compare_vector(actual["ema_teacher"], expected_ema, "EMA")

    if not math.isclose(
        float(actual["loss"]),
        expected_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("loss mismatch")

    print("Module 28 Python ↔ C++ DINO parity: PASS")


if __name__ == "__main__":
    main()
