"""Verify Module 06 Python inference against the native C++ runtime."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from linear_classifier import linear_logit, load_model, predict, predict_probability


SAMPLE = [0.5, 0.2, 0.7, 0.1]


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
        "logit": linear_logit(SAMPLE, model.weights, model.bias),
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
        raise AssertionError(
            f"prediction: C++={actual['prediction']} Python={expected['prediction']}"
        )

    print("Module 06 Python ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
