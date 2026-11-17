"""Verify Module 11 PyTorch, ONNX Runtime and C++17 parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from transfer_learning import (
    TinyTransferCNN,
    apply_transfer_artifact,
    export_onnx,
    load_transfer_artifact,
    onnx_runtime_logits,
    vertical_example_tensor,
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

    artifact = load_transfer_artifact(args.artifact)
    model = TinyTransferCNN()
    apply_transfer_artifact(model, artifact)
    model.eval()

    sample = vertical_example_tensor()
    with torch.no_grad():
        pytorch_logits = model(sample).squeeze(0).double().tolist()

    with tempfile.TemporaryDirectory() as directory:
        onnx_path = Path(directory) / "transfer.onnx"
        export_onnx(model, onnx_path)
        ort_logits = (
            onnx_runtime_logits(onnx_path, sample)
            .reshape(-1)
            .astype(float)
            .tolist()
        )

    native = run_native(args.executable, args.artifact)
    cpp_logits = native["logits"]

    for name, values in (
        ("onnxruntime", ort_logits),
        ("cpp", cpp_logits),
    ):
        if len(values) != len(pytorch_logits):
            raise AssertionError(f"{name} logit count mismatch")
        for index, (actual, expected) in enumerate(
            zip(values, pytorch_logits, strict=True)
        ):
            if not math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    f"{name}[{index}]={actual} PyTorch={expected}"
                )

    print("Module 11 PyTorch ↔ ONNX Runtime ↔ C++ parity: PASS")


if __name__ == "__main__":
    main()
