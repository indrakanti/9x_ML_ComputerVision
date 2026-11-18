"""Module 27 Python ↔ C++ masked-modeling parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from masked_image_modeling import (
    load_artifact,
    masked_mse,
    reconstruct_masked,
)


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
    expected_reconstruction = reconstruct_masked(artifact)
    expected_loss = masked_mse(artifact)
    actual = run_native(args.executable, args.artifact)

    for row_cpp, row_py in zip(
        actual["reconstruction"],
        expected_reconstruction,
        strict=True,
    ):
        for cpp, py in zip(row_cpp, row_py, strict=True):
            if not math.isclose(
                float(cpp),
                py,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError("reconstruction mismatch")

    if not math.isclose(
        float(actual["loss"]),
        expected_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("masked loss mismatch")

    print("Module 27 Python ↔ C++ masked-modeling parity: PASS")


if __name__ == "__main__":
    main()
