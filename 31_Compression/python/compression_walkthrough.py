"""Walkthrough for Module 31."""

from __future__ import annotations

from pathlib import Path

from compression import (
    distillation_loss,
    load_artifact,
    prune_weights,
    quantize_symmetric_int8,
    sparsity,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "compression.txt")

    pruned = prune_weights(
        artifact.weights,
        artifact.prune_threshold,
    )
    scale, quantized, dequantized, max_error = quantize_symmetric_int8(
        artifact.quant_values
    )

    print("Distillation loss:", distillation_loss(artifact))
    print("Pruned weights:", pruned)
    print("Sparsity:", sparsity(pruned))
    print("INT8 scale:", scale)
    print("Quantized:", quantized)
    print("Dequantized:", dequantized)
    print("Max error:", max_error)


if __name__ == "__main__":
    main()
