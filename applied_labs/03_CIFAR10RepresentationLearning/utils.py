"""Shared runtime helpers for Applied Lab 03."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data import basic_transform
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(dataset: Dataset[Any], maximum: int | None) -> Dataset[Any]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def cifar_dataset(
    root: Path,
    train: bool,
    download: bool,
    augment: bool,
) -> Dataset[tuple[Tensor, int]]:
    return CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=basic_transform(augment),
    )


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def split_indices(
    length: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_count = max(1, int(round(length * val_fraction)))
    return indices[val_count:], indices[:val_count]
