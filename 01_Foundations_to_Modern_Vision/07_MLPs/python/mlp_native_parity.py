"""Verify Module 07 Python MLP inference against C++17."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from mlp import forward_logit, load_model, predict, predict_probability

SAMPLE = [0.6, 0.2, 0.8, 0.3]


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

    model = load_model(args.artifact)
    expected = {
        "logit": forward_logit(model, SAMPLE),
        "probability": predict_probability(model, SAMPLE),
        "prediction": predict(model, SAMPLE),
    }
    actual = run_native(args.executable, args.artifact)

    for key in ("logit", "probability"):
        if not math.isclose(
            float(actual[key]),
            float(expected[key]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{key}: C++={actual[key]} Python={expected[key]}")
    if int(actual["prediction"]) != int(expected["prediction"]):
        raise AssertionError("prediction mismatch")

    print("Module 07 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
