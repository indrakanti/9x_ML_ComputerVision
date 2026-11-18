from __future__ import annotations

from pathlib import Path

import pytest
from masked_image_modeling import (
    load_artifact,
    masked_mse,
    patchify,
    pytorch_reconstruction_and_loss,
    reconstruct_masked,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "mim.txt"


def test_patchify_produces_four_patches() -> None:
    artifact = load_artifact(artifact_path())

    assert len(patchify(artifact)) == 4


def test_manual_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    manual_reconstruction = reconstruct_masked(artifact)
    manual_loss = masked_mse(artifact)
    torch_reconstruction, torch_loss = pytorch_reconstruction_and_loss(
        artifact
    )

    assert torch_reconstruction.tolist() == pytest.approx(
        manual_reconstruction
    )
    assert float(torch_loss.item()) == pytest.approx(manual_loss)
