"""CIFAR-10 domain helpers for Applied Lab 04."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def transform(domain: str, train: bool) -> transforms.Compose:
    steps: list[Any] = []
    if train:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if domain == "grayscale":
        steps.append(transforms.Grayscale(num_output_channels=3))
    elif domain != "standard":
        raise ValueError(f"unsupported domain: {domain}")

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return transforms.Compose(steps)


def cifar(
    root: Path,
    train: bool,
    download: bool,
    domain: str,
    augment: bool,
) -> Dataset[tuple[Tensor, int]]:
    return CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform(domain, augment),
    )


def limited(dataset: Dataset[Any], maximum: int | None) -> Dataset[Any]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))
