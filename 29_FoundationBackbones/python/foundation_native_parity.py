"""Module 29 Python ↔ C++ foundation-adapter parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from foundation_backbone import (
    adapter_forward,
    load_artifact,
    total_adapter_parameters,
)


def run_native(executable: Path, artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _compare(
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
    expected_classification = adapter_forward(
        artifact.features,
        artifact.classification,
    )
    expected_geometry = adapter_forward(
        artifact.features,
        artifact.geometry,
    )
    expected_params = total_adapter_parameters(artifact)
    actual = run_native(args.executable, args.artifact)

    if actual["feature_revision"] != artifact.feature_revision:
        raise AssertionError("feature revision mismatch")

    _compare(
        actual["classification"],
        expected_classification,
        "classification",
    )
    _compare(actual["geometry"], expected_geometry, "geometry")

    if int(actual["adapter_parameters"]) != expected_params:
        raise AssertionError("adapter parameter count mismatch")

    print("Module 29 Python ↔ C++ foundation-adapter parity: PASS")


if __name__ == "__main__":
    main()
