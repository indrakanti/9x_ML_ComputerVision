from __future__ import annotations

from pathlib import Path

import pytest
import torch
from training_robustness import (
    AugmentationConfig,
    DeploymentManifest,
    RegularizedTinyCNN,
    augment_batch,
    build_optimizer_and_scheduler,
    load_deployment_manifest,
    load_training_checkpoint,
    save_best_if_improved,
    save_training_checkpoint,
    write_deployment_manifest,
)


def test_augmentation_is_deterministic_with_same_seed() -> None:
    images = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    config = AugmentationConfig(noise_amplitude=0.1)

    first = augment_batch(
        images,
        config,
        torch.Generator().manual_seed(7),
    )
    second = augment_batch(
        images,
        config,
        torch.Generator().manual_seed(7),
    )
    assert torch.equal(first, second)


def test_scheduler_decays_learning_rate() -> None:
    model = RegularizedTinyCNN()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        learning_rate=0.2,
        step_size=2,
        gamma=0.5,
    )
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_checkpoint_restores_optimizer_and_scheduler(tmp_path: Path) -> None:
    model = RegularizedTinyCNN()
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    optimizer.step()
    scheduler.step()
    path = tmp_path / "latest.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        best_accuracy=0.8,
        config={"seed": 7, "augmentation": "v1"},
    )

    restored = RegularizedTinyCNN()
    restored_optimizer, restored_scheduler = build_optimizer_and_scheduler(
        restored
    )
    epoch, best_accuracy, config = load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert epoch == 4
    assert best_accuracy == pytest.approx(0.8)
    assert config["seed"] == 7
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_best_checkpoint_only_updates_when_metric_improves(tmp_path: Path) -> None:
    model = RegularizedTinyCNN()
    path = tmp_path / "best.pt"

    best = save_best_if_improved(
        path,
        accuracy=0.75,
        best_accuracy=0.7,
        model=model,
    )
    assert best == pytest.approx(0.75)
    before = path.read_bytes()

    best = save_best_if_improved(
        path,
        accuracy=0.72,
        best_accuracy=best,
        model=model,
    )
    assert best == pytest.approx(0.75)
    assert path.read_bytes() == before


def test_dropout_changes_training_behavior_but_not_eval_behavior() -> None:
    torch.manual_seed(7)
    model = RegularizedTinyCNN(dropout_probability=0.5)
    sample = torch.ones((1, 1, 4, 4), dtype=torch.float32)

    model.eval()
    first_eval = model(sample)
    second_eval = model(sample)
    assert torch.equal(first_eval, second_eval)


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.txt"
    manifest = DeploymentManifest(
        version=1,
        selected_checkpoint="best",
        dropout_inference="disabled",
        augmentation_inference="disabled",
        preprocessing_revision="synthetic-v1",
        class_count=2,
    )
    write_deployment_manifest(path, manifest)
    assert load_deployment_manifest(path) == manifest


def test_manifest_rejects_training_behavior_at_inference(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(
        "version=1\n"
        "selected_checkpoint=best\n"
        "dropout_inference=enabled\n"
        "augmentation_inference=disabled\n"
        "preprocessing_revision=synthetic-v1\n"
        "class_count=2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_deployment_manifest(path)
