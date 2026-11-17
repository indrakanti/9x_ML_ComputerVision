"""Tiny CNN training lifecycle for Module 09."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.relu(self.conv(inputs))
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def make_synthetic_dataset(repeats: int = 16) -> tuple[Tensor, Tensor]:
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    vertical = torch.zeros((1, 4, 4), dtype=torch.float32)
    vertical[:, :, 1:3] = 1.0

    horizontal = torch.zeros((1, 4, 4), dtype=torch.float32)
    horizontal[:, 1:3, :] = 1.0

    images: list[Tensor] = []
    labels: list[int] = []
    for _ in range(repeats):
        images.extend((vertical.clone(), horizontal.clone()))
        labels.extend((0, 1))

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def train_one_epoch(
    model: TinyCNN,
    images: Tensor,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
) -> EpochMetrics:
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    return EpochMetrics(loss=float(loss.item()), accuracy=accuracy)


def evaluate(
    model: TinyCNN,
    images: Tensor,
    labels: Tensor,
    criterion: nn.CrossEntropyLoss,
) -> EpochMetrics:
    model.eval()
    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        accuracy = float((predictions == labels).float().mean().item())
    return EpochMetrics(loss=float(loss.item()), accuracy=accuracy)


def train_model(
    *,
    epochs: int = 120,
    learning_rate: float = 0.2,
    seed: int = 7,
) -> tuple[TinyCNN, list[EpochMetrics]]:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")

    torch.manual_seed(seed)
    model = TinyCNN()
    images, labels = make_synthetic_dataset()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[EpochMetrics] = []
    for _ in range(epochs):
        train_one_epoch(model, images, labels, optimizer, criterion)
        history.append(evaluate(model, images, labels, criterion))
    return model, history


def save_checkpoint(
    model: TinyCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, model: TinyCNN, optimizer: torch.optim.Optimizer) -> int:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"])


def export_native_artifact(model: TinyCNN, path: Path) -> None:
    if model.conv.bias is None or model.classifier.bias is None:
        raise ValueError("native artifact export requires bias-enabled layers")

    conv_w = model.conv.weight.detach().cpu().double().reshape(-1).tolist()
    conv_b = model.conv.bias.detach().cpu().double().tolist()
    fc_w = model.classifier.weight.detach().cpu().double().reshape(-1).tolist()
    fc_b = model.classifier.bias.detach().cpu().double().tolist()

    lines = [
        "version=1",
        "input_h=4",
        "input_w=4",
        "conv_out=2",
        "kernel=3",
        "padding=1",
        "conv_weights=" + ",".join(f"{value:.17g}" for value in conv_w),
        "conv_bias=" + ",".join(f"{value:.17g}" for value in conv_b),
        "fc_weights=" + ",".join(f"{value:.17g}" for value in fc_w),
        "fc_bias=" + ",".join(f"{value:.17g}" for value in fc_b),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_native_artifact(path: Path) -> dict[str, list[float] | int]:
    raw: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition("=")
        if sep != "=":
            raise ValueError(f"invalid artifact line: {line}")
        raw[key] = value

    required = {
        "version",
        "input_h",
        "input_w",
        "conv_out",
        "kernel",
        "padding",
        "conv_weights",
        "conv_bias",
        "fc_weights",
        "fc_bias",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"missing artifact fields: {sorted(missing)}")

    def vector(name: str) -> list[float]:
        return [float(item) for item in raw[name].split(",") if item]

    return {
        "version": int(raw["version"]),
        "input_h": int(raw["input_h"]),
        "input_w": int(raw["input_w"]),
        "conv_out": int(raw["conv_out"]),
        "kernel": int(raw["kernel"]),
        "padding": int(raw["padding"]),
        "conv_weights": vector("conv_weights"),
        "conv_bias": vector("conv_bias"),
        "fc_weights": vector("fc_weights"),
        "fc_bias": vector("fc_bias"),
    }


def artifact_inference(artifact: dict[str, list[float] | int], image: list[float]) -> list[float]:
    input_h = cast(int, artifact["input_h"])
    input_w = cast(int, artifact["input_w"])
    conv_out = cast(int, artifact["conv_out"])
    kernel = cast(int, artifact["kernel"])
    padding = cast(int, artifact["padding"])
    conv_weights = cast(list[float], artifact["conv_weights"])
    conv_bias = cast(list[float], artifact["conv_bias"])
    fc_weights = cast(list[float], artifact["fc_weights"])
    fc_bias = cast(list[float], artifact["fc_bias"])

    if len(image) != input_h * input_w:
        raise ValueError("image size mismatch")
    if len(conv_weights) != conv_out * kernel * kernel:
        raise ValueError("conv weight count mismatch")
    if len(conv_bias) != conv_out:
        raise ValueError("conv bias count mismatch")
    if len(fc_weights) != 2 * conv_out or len(fc_bias) != 2:
        raise ValueError("classifier shape mismatch")

    pooled: list[float] = []
    for out_channel in range(conv_out):
        total = 0.0
        for y in range(input_h):
            for x in range(input_w):
                value = conv_bias[out_channel]
                for ky in range(kernel):
                    for kx in range(kernel):
                        iy = y + ky - padding
                        ix = x + kx - padding
                        if 0 <= iy < input_h and 0 <= ix < input_w:
                            weight_index = (
                                out_channel * kernel * kernel
                                + ky * kernel
                                + kx
                            )
                            value += image[iy * input_w + ix] * conv_weights[weight_index]
                total += max(0.0, value)
        pooled.append(total / (input_h * input_w))

    return [
        fc_bias[output]
        + sum(
            pooled[channel] * fc_weights[output * conv_out + channel]
            for channel in range(conv_out)
        )
        for output in range(2)
    ]


def vertical_example() -> list[float]:
    return [
        0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
    ]
