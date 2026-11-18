"""Module 30 Python ↔ C++ PEFT parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from peft import (
    adapter_forward,
    load_artifact,
    lora_forward,
    parameter_counts,
    probe_forward,
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
    actual = run_native(args.executable, args.artifact)

    _compare(actual["probe"], probe_forward(artifact), "probe")
    _compare(actual["adapter"], adapter_forward(artifact), "adapter")
    _compare(actual["lora"], lora_forward(artifact), "lora")

    expected_counts = parameter_counts(artifact)
    if actual["parameter_counts"] != expected_counts:
        raise AssertionError("parameter-count mismatch")

    print("Module 30 Python ↔ C++ PEFT parity: PASS")


if __name__ == "__main__":
    main()
