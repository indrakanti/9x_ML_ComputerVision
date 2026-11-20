"""Module 20 Python ↔ C++ proposal parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from two_stage import load_artifact, proposals


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

    expected = proposals(load_artifact(args.artifact))
    actual = run_native(args.executable, args.artifact)["proposals"]

    if len(actual) != len(expected):
        raise AssertionError("proposal count mismatch")

    for index, (cpp, python) in enumerate(zip(actual, expected, strict=True)):
        box, score = python
        for key, expected_value in (
            ("x1", box.x1),
            ("y1", box.y1),
            ("x2", box.x2),
            ("y2", box.y2),
            ("score", score),
        ):
            if not math.isclose(
                float(cpp[key]),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"{key} mismatch at proposal {index}")

    print("Module 20 Python ↔ C++ proposal parity: PASS")


if __name__ == "__main__":
    main()
