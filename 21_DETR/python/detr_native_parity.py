"""Module 21 Python ↔ C++ matching parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from detr_matching import cost_matrix, load_artifact, optimal_assignment


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
    matrix = cost_matrix(artifact)
    expected_assignment = optimal_assignment(matrix)
    actual = run_native(args.executable, args.artifact)

    actual_matrix = actual["costs"]
    for row_cpp, row_py in zip(actual_matrix, matrix, strict=True):
        for value_cpp, value_py in zip(row_cpp, row_py, strict=True):
            if not math.isclose(
                float(value_cpp),
                value_py,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError("cost-matrix mismatch")

    actual_assignment = [
        (int(item[0]), int(item[1]))
        for item in actual["assignment"]
    ]
    if actual_assignment != expected_assignment:
        raise AssertionError(
            f"assignment mismatch C++={actual_assignment} "
            f"Python={expected_assignment}"
        )

    print("Module 21 Python ↔ C++ DETR matching parity: PASS")


if __name__ == "__main__":
    main()
