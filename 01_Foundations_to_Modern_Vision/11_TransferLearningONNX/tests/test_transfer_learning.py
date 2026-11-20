from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transfer_learning import (
    TinyTransferCNN,
    apply_backbone_artifact,
    apply_transfer_artifact,
    export_onnx,
    freeze_backbone,
    load_backbone_artifact,
    load_transfer_artifact,
    make_transfer_dataset,
    onnx_runtime_logits,
    train_classifier_head,
    vertical_example_tensor,
)


def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_freeze_backbone_disables_gradients() -> None:
    model = TinyTransferCNN()
    freeze_backbone(model)
    assert all(
        not parameter.requires_grad
        for parameter in model.backbone.parameters()
    )


def test_head_training_reduces_loss_without_changing_backbone() -> None:
    torch.manual_seed(7)
    model = TinyTransferCNN()
    artifact = load_backbone_artifact(
        module_root() / "artifacts" / "pretrained_backbone.txt"
    )
    apply_backbone_artifact(model, artifact)

    before = [
        parameter.detach().clone()
        for parameter in model.backbone.parameters()
    ]
    images, labels = make_transfer_dataset()
    losses = train_classifier_head(model, images, labels)

    assert losses[-1] < losses[0]
    for expected, actual in zip(
        before,
        model.backbone.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual.detach())


def test_transfer_artifact_matches_onnx_runtime(tmp_path: Path) -> None:
    artifact = load_transfer_artifact(
        module_root() / "artifacts" / "transfer_model_smoke.txt"
    )
    model = TinyTransferCNN()
    apply_transfer_artifact(model, artifact)
    model.eval()

    sample = vertical_example_tensor()
    with torch.no_grad():
        expected = model(sample).numpy()

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, onnx_path)
    actual = onnx_runtime_logits(onnx_path, sample)

    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_exported_onnx_exists_and_is_nonempty(tmp_path: Path) -> None:
    artifact = load_transfer_artifact(
        module_root() / "artifacts" / "transfer_model_smoke.txt"
    )
    model = TinyTransferCNN()
    apply_transfer_artifact(model, artifact)

    path = tmp_path / "model.onnx"
    export_onnx(model, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_dataset_shapes() -> None:
    images, labels = make_transfer_dataset(repeats=3)
    assert images.shape == (6, 1, 4, 4)
    assert labels.shape == (6,)


def test_invalid_dataset_repeat_count_rejected() -> None:
    with pytest.raises(ValueError):
        make_transfer_dataset(0)


def test_onnx_output_has_expected_shape(tmp_path: Path) -> None:
    artifact = load_transfer_artifact(
        module_root() / "artifacts" / "transfer_model_smoke.txt"
    )
    model = TinyTransferCNN()
    apply_transfer_artifact(model, artifact)
    path = tmp_path / "model.onnx"
    export_onnx(model, path)

    output = onnx_runtime_logits(path, vertical_example_tensor())
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)
