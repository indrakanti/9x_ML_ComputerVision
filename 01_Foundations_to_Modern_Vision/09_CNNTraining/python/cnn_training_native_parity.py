"""Verify Module 09 Python artifact inference against C++17."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from cnn_training import artifact_inference, load_native_artifact, vertical_example


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

    artifact = load_native_artifact(args.artifact)
    expected = artifact_inference(artifact, vertical_example())
    actual = run_native(args.executable, args.artifact)

    logits = actual["logits"]
    if len(logits) != len(expected):
        raise AssertionError("logit count mismatch")

    for index, (cpp_value, python_value) in enumerate(zip(logits, expected, strict=True)):
        if not math.isclose(
            float(cpp_value),
            float(python_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"logit[{index}]: C++={cpp_value} Python={python_value}"
            )

    print("Module 09 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
