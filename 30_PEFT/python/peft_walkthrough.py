"""Walkthrough for Module 30."""

from __future__ import annotations

from pathlib import Path

from peft import (
    adapter_forward,
    load_artifact,
    lora_forward,
    parameter_counts,
    probe_forward,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "peft.txt")

    print("Probe:", probe_forward(artifact))
    print("Adapter:", adapter_forward(artifact))
    print("LoRA:", lora_forward(artifact))
    print("Parameter counts:", parameter_counts(artifact))


if __name__ == "__main__":
    main()
