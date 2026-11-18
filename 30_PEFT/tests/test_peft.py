from __future__ import annotations

from pathlib import Path

import pytest
from peft import (
    load_artifact,
    lora_forward,
    parameter_counts,
    pytorch_lora_forward,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "peft.txt"


def test_lora_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    manual = lora_forward(artifact)
    torch_output = pytorch_lora_forward(artifact).tolist()

    assert torch_output == pytest.approx(manual, rel=1e-12, abs=1e-12)


def test_lora_uses_fewer_trainable_parameters_than_full_layer() -> None:
    artifact = load_artifact(artifact_path())
    counts = parameter_counts(artifact)

    assert counts["lora"] < counts["full_linear"]
