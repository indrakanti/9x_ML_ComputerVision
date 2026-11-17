from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cnn_training import (
    TinyCNN,
    artifact_inference,
    evaluate,
    export_native_artifact,
    load_checkpoint,
    load_native_artifact,
    make_synthetic_dataset,
    save_checkpoint,
    train_model,
    vertical_example,
)


def test_dataset_shapes() -> None:
    images, labels = make_synthetic_dataset(repeats=3)
    assert images.shape == (6, 1, 4, 4)
    assert labels.shape == (6,)


def test_training_learns_synthetic_task() -> None:
    _, history = train_model(epochs=120, learning_rate=0.2, seed=7)
    assert history[-1].loss < history[0].loss
    assert history[-1].accuracy >= 0.95


def test_evaluate_does_not_create_parameter_gradients() -> None:
    model = TinyCNN()
    images, labels = make_synthetic_dataset(repeats=2)
    metrics = evaluate(model, images, labels, torch.nn.CrossEntropyLoss())

    assert 0.0 <= metrics.accuracy <= 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = TinyCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(model, optimizer, 5, path)

    restored = TinyCNN()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    epoch = load_checkpoint(path, restored, restored_optimizer)

    assert epoch == 5
    for expected, actual in zip(
        model.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_native_artifact_matches_pytorch(tmp_path: Path) -> None:
    model, _ = train_model(epochs=120, learning_rate=0.2, seed=7)
    artifact_path = tmp_path / "model.txt"
    export_native_artifact(model, artifact_path)

    artifact = load_native_artifact(artifact_path)
    expected = artifact_inference(artifact, vertical_example())

    image = torch.tensor(vertical_example(), dtype=torch.float32).reshape(1, 1, 4, 4)
    model.eval()
    with torch.no_grad():
        actual = model(image).squeeze(0).double().tolist()

    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_invalid_repeat_count_rejected() -> None:
    with pytest.raises(ValueError):
        make_synthetic_dataset(0)
