"""Module 31 Python ↔ C++ compression parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from compression import (
    distillation_loss,
    load_artifact,
    prune_weights,
    quantize_symmetric_int8,
    sparsity,
)


def run_native(executable: Path, artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _compare_float_lists(
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
    expected_pruned = prune_weights(
        artifact.weights,
        artifact.prune_threshold,
    )
    expected_sparsity = sparsity(expected_pruned)
    (
        expected_scale,
        expected_quantized,
        expected_dequantized,
        expected_error,
    ) = quantize_symmetric_int8(artifact.quant_values)
    expected_loss = distillation_loss(artifact)

    actual = run_native(args.executable, args.artifact)

    if not math.isclose(
        float(actual["distillation_loss"]),
        expected_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("distillation loss mismatch")

    _compare_float_lists(
        actual["pruned"],
        expected_pruned,
        "pruned weights",
    )

    if not math.isclose(
        float(actual["sparsity"]),
        expected_sparsity,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("sparsity mismatch")

    if not math.isclose(
        float(actual["scale"]),
        expected_scale,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("quantization scale mismatch")

    if [int(value) for value in actual["quantized"]] != expected_quantized:
        raise AssertionError("quantized INT8 mismatch")

    _compare_float_lists(
        actual["dequantized"],
        expected_dequantized,
        "dequantized values",
    )

    if not math.isclose(
        float(actual["max_error"]),
        expected_error,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("quantization error mismatch")

    print("Module 31 Python ↔ C++ compression parity: PASS")


if __name__ == "__main__":
    main()
