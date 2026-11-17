"""Transfer learning and ONNX export for Module 11."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BackboneArtifact:
    conv_out: int
    kernel: int
    padding: int
    conv_weights: tuple[float, ...]
    conv_bias: tuple[float, ...]


@dataclass(frozen=True)
class TransferArtifact:
    input_h: int
    input_w: int
    conv_out: int
    kernel: int
    padding: int
    conv_weights: tuple[float, ...]
    conv_bias: tuple[float, ...]
    fc_weights: tuple[float, ...]
    fc_bias: tuple[float, ...]
    class_names: tuple[str, ...]


class TinyTransferCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.relu(self.backbone(inputs))
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def _parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator != "=":
            raise ValueError(f"invalid artifact line: {raw}")
        values[key] = value
    return values


def _float_vector(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item)


def load_backbone_artifact(path: Path) -> BackboneArtifact:
    values = _parse_key_values(path)
    if int(values["version"]) != 1:
        raise ValueError("unsupported backbone artifact version")

    artifact = BackboneArtifact(
        conv_out=int(values["conv_out"]),
        kernel=int(values["kernel"]),
        padding=int(values["padding"]),
        conv_weights=_float_vector(values["conv_weights"]),
        conv_bias=_float_vector(values["conv_bias"]),
    )
    expected = artifact.conv_out * artifact.kernel * artifact.kernel
    if len(artifact.conv_weights) != expected:
        raise ValueError("backbone weight count mismatch")
    if len(artifact.conv_bias) != artifact.conv_out:
        raise ValueError("backbone bias count mismatch")
    return artifact


def load_transfer_artifact(path: Path) -> TransferArtifact:
    values = _parse_key_values(path)
    if int(values["version"]) != 1:
        raise ValueError("unsupported transfer artifact version")

    artifact = TransferArtifact(
        input_h=int(values["input_h"]),
        input_w=int(values["input_w"]),
        conv_out=int(values["conv_out"]),
        kernel=int(values["kernel"]),
        padding=int(values["padding"]),
        conv_weights=_float_vector(values["conv_weights"]),
        conv_bias=_float_vector(values["conv_bias"]),
        fc_weights=_float_vector(values["fc_weights"]),
        fc_bias=_float_vector(values["fc_bias"]),
        class_names=tuple(values["class_names"].split(",")),
    )

    if len(artifact.conv_weights) != (
        artifact.conv_out * artifact.kernel * artifact.kernel
    ):
        raise ValueError("conv weight count mismatch")
    if len(artifact.conv_bias) != artifact.conv_out:
        raise ValueError("conv bias count mismatch")
    if len(artifact.fc_weights) != len(artifact.class_names) * artifact.conv_out:
        raise ValueError("classifier weight count mismatch")
    if len(artifact.fc_bias) != len(artifact.class_names):
        raise ValueError("classifier bias count mismatch")
    return artifact


def apply_backbone_artifact(
    model: TinyTransferCNN,
    artifact: BackboneArtifact,
) -> None:
    if model.backbone.bias is None:
        raise ValueError("backbone must be bias-enabled")

    weights = torch.tensor(
        artifact.conv_weights,
        dtype=torch.float32,
    ).reshape(artifact.conv_out, 1, artifact.kernel, artifact.kernel)
    bias = torch.tensor(artifact.conv_bias, dtype=torch.float32)

    with torch.no_grad():
        model.backbone.weight.copy_(weights)
        model.backbone.bias.copy_(bias)


def apply_transfer_artifact(
    model: TinyTransferCNN,
    artifact: TransferArtifact,
) -> None:
    if model.backbone.bias is None or model.classifier.bias is None:
        raise ValueError("transfer model requires bias-enabled layers")

    with torch.no_grad():
        model.backbone.weight.copy_(
            torch.tensor(
                artifact.conv_weights,
                dtype=torch.float32,
            ).reshape(
                artifact.conv_out,
                1,
                artifact.kernel,
                artifact.kernel,
            )
        )
        model.backbone.bias.copy_(
            torch.tensor(artifact.conv_bias, dtype=torch.float32)
        )
        model.classifier.weight.copy_(
            torch.tensor(
                artifact.fc_weights,
                dtype=torch.float32,
            ).reshape(len(artifact.class_names), artifact.conv_out)
        )
        model.classifier.bias.copy_(
            torch.tensor(artifact.fc_bias, dtype=torch.float32)
        )


def freeze_backbone(model: TinyTransferCNN) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def make_transfer_dataset(repeats: int = 16) -> tuple[Tensor, Tensor]:
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


def train_classifier_head(
    model: TinyTransferCNN,
    images: Tensor,
    labels: Tensor,
    *,
    epochs: int = 80,
    learning_rate: float = 0.4,
) -> list[float]:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    freeze_backbone(model)
    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def export_onnx(
    model: TinyTransferCNN,
    path: Path,
) -> None:
    model.eval()
    example = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


def onnx_runtime_logits(path: Path, inputs: Tensor) -> np.ndarray:
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    input_array = inputs.detach().cpu().numpy().astype(np.float32)
    outputs = session.run(["logits"], {"input": input_array})
    return cast(np.ndarray, outputs[0])


def vertical_example_tensor() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
