"""Training robustness utilities for Module 10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.25
    vertical_flip_probability: float = 0.25
    noise_amplitude: float = 0.05


@dataclass(frozen=True)
class DeploymentManifest:
    version: int
    selected_checkpoint: str
    dropout_inference: str
    augmentation_inference: str
    preprocessing_revision: str
    class_count: int


class RegularizedTinyCNN(nn.Module):
    def __init__(self, dropout_probability: float = 0.25) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.relu(self.conv(inputs))
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(pooled))


def validate_augmentation_config(config: AugmentationConfig) -> None:
    for probability in (
        config.horizontal_flip_probability,
        config.vertical_flip_probability,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("flip probabilities must be in [0, 1]")
    if config.noise_amplitude < 0.0:
        raise ValueError("noise amplitude must be >= 0")


def augment_batch(
    images: Tensor,
    config: AugmentationConfig,
    generator: torch.Generator,
) -> Tensor:
    validate_augmentation_config(config)
    if images.ndim != 4:
        raise ValueError("images must be NCHW")

    output = images.clone()
    for index in range(output.shape[0]):
        horizontal_sample = torch.rand((), generator=generator).item()
        if horizontal_sample < config.horizontal_flip_probability:
            output[index] = torch.flip(output[index], dims=(2,))
        vertical_sample = torch.rand((), generator=generator).item()
        if vertical_sample < config.vertical_flip_probability:
            output[index] = torch.flip(output[index], dims=(1,))

    if config.noise_amplitude > 0.0:
        noise = torch.rand(
            output.shape,
            generator=generator,
            dtype=output.dtype,
            device=output.device,
        )
        noise = (noise * 2.0 - 1.0) * config.noise_amplitude
        output = output + noise
    return output


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-4,
    step_size: int = 10,
    gamma: float = 0.5,
) -> tuple[torch.optim.SGD, torch.optim.lr_scheduler.StepLR]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
    return optimizer, scheduler


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_accuracy: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float, dict[str, Any]]:
    payload: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    return (
        int(payload["epoch"]),
        float(payload["best_accuracy"]),
        dict(payload["config"]),
    )


def save_best_if_improved(
    path: Path,
    *,
    accuracy: float,
    best_accuracy: float,
    model: nn.Module,
) -> float:
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), path)
        return accuracy
    return best_accuracy


def write_deployment_manifest(
    path: Path,
    manifest: DeploymentManifest,
) -> None:
    lines = [
        f"version={manifest.version}",
        f"selected_checkpoint={manifest.selected_checkpoint}",
        f"dropout_inference={manifest.dropout_inference}",
        f"augmentation_inference={manifest.augmentation_inference}",
        f"preprocessing_revision={manifest.preprocessing_revision}",
        f"class_count={manifest.class_count}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_deployment_manifest(path: Path) -> DeploymentManifest:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator != "=":
            raise ValueError(f"invalid manifest line: {raw}")
        values[key] = value

    required = {
        "version",
        "selected_checkpoint",
        "dropout_inference",
        "augmentation_inference",
        "preprocessing_revision",
        "class_count",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing manifest fields: {sorted(missing)}")

    manifest = DeploymentManifest(
        version=int(values["version"]),
        selected_checkpoint=values["selected_checkpoint"],
        dropout_inference=values["dropout_inference"],
        augmentation_inference=values["augmentation_inference"],
        preprocessing_revision=values["preprocessing_revision"],
        class_count=int(values["class_count"]),
    )
    validate_deployment_manifest(manifest)
    return manifest


def validate_deployment_manifest(manifest: DeploymentManifest) -> None:
    if manifest.version != 1:
        raise ValueError("unsupported manifest version")
    if manifest.selected_checkpoint not in {"best", "latest"}:
        raise ValueError("selected_checkpoint must be best or latest")
    if manifest.dropout_inference != "disabled":
        raise ValueError("dropout must be disabled for inference")
    if manifest.augmentation_inference != "disabled":
        raise ValueError("augmentation must be disabled for inference")
    if not manifest.preprocessing_revision:
        raise ValueError("preprocessing revision must be non-empty")
    if manifest.class_count <= 0:
        raise ValueError("class_count must be > 0")
