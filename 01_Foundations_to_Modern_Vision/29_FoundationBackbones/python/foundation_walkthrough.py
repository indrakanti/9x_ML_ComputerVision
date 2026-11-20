"""Walkthrough for Module 29."""

from __future__ import annotations

from pathlib import Path

from foundation_backbone import (
    adapter_forward,
    adapter_overhead_ratio,
    load_artifact,
    total_adapter_parameters,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "foundation.txt")

    print("Feature revision:", artifact.feature_revision)
    print(
        "Classification logits:",
        adapter_forward(
            artifact.features,
            artifact.classification,
        ),
    )
    print(
        "Geometry output:",
        adapter_forward(
            artifact.features,
            artifact.geometry,
        ),
    )
    print("Adapter parameters:", total_adapter_parameters(artifact))
    print("Adapter/backbone ratio:", adapter_overhead_ratio(artifact))


if __name__ == "__main__":
    main()
