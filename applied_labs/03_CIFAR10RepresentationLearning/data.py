"""CIFAR-10 data helpers for Applied Lab 03."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def basic_transform(train: bool) -> transforms.Compose:
    steps: list[object] = []
    if train:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return transforms.Compose(steps)


def dino_view_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                32,
                scale=(0.5, 1.0),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


class TwoViewCIFAR(Dataset[tuple[Tensor, Tensor, int]]):
    def __init__(
        self,
        root: Path,
        train: bool,
        download: bool,
    ) -> None:
        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=download,
        )
        self.view = dino_view_transform()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        image, label = self.dataset[index]
        if not isinstance(image, Image.Image):
            raise TypeError("expected PIL image before DINO transforms")
        return self.view(image), self.view(image), int(label)


def limited(dataset: Dataset[object], maximum: int | None) -> Dataset[object]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def prompt_texts() -> tuple[str, ...]:
    return tuple(f"a photo of a {name}" for name in CIFAR_CLASSES)


def character_vocabulary() -> dict[str, int]:
    characters = sorted(set("".join(prompt_texts())))
    return {character: index + 1 for index, character in enumerate(characters)}


def encode_prompts(max_length: int = 32) -> Tensor:
    vocab = character_vocabulary()
    rows: list[list[int]] = []
    for text in prompt_texts():
        encoded = [vocab[character] for character in text[:max_length]]
        encoded.extend([0] * (max_length - len(encoded)))
        rows.append(encoded)
    return torch.tensor(rows, dtype=torch.long)
