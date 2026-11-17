"""Module 17 Python ↔ C++ scorecard parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from backbone_eval import load, rank


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

    weights, models = load(args.artifact)
    expected = rank(weights, models)
    actual = run_native(args.executable, args.artifact)["ranking"]

    if [row["name"] for row in actual] != [name for name, _ in expected]:
        raise AssertionError("ranking order mismatch")

    for row, (_, score) in zip(actual, expected, strict=True):
        if not math.isclose(
            float(row["score"]),
            score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError("score mismatch")

    print("Module 17 Python ↔ C++ scorecard parity: PASS")


if __name__ == "__main__":
    main()
