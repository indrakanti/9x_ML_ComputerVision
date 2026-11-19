"""Oxford-IIIT Pet dataset utilities for Applied Lab 02."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGE_SIZE = 128
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PetTarget:
    box_xyxy: Tensor
    mask: Tensor


def trimap_to_binary(mask: Tensor) -> Tensor:
    """Map Oxford trimap 1/2/3 to binary pet/background."""
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return (mask != 2).to(torch.float32)


def mask_to_box(mask: Tensor) -> Tensor:
    """Return normalized xyxy box around foreground pixels."""
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    locations = torch.nonzero(mask > 0.5, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError("foreground mask is empty")

    y_min = int(locations[:, 0].min().item())
    y_max = int(locations[:, 0].max().item()) + 1
    x_min = int(locations[:, 1].min().item())
    x_max = int(locations[:, 1].max().item()) + 1
    height, width = mask.shape

    return torch.tensor(
        [
            x_min / width,
            y_min / height,
            x_max / width,
            y_max / height,
        ],
        dtype=torch.float32,
    )


def prepare_sample(
    image: Image.Image,
    trimap: Image.Image,
    image_size: int = IMAGE_SIZE,
    horizontal_flip: bool = False,
) -> tuple[Tensor, PetTarget]:
    image = TF.resize(
        image,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    trimap = TF.resize(
        trimap,
        [image_size, image_size],
        interpolation=InterpolationMode.NEAREST,
    )

    if horizontal_flip:
        image = TF.hflip(image)
        trimap = TF.hflip(trimap)

    image_tensor = TF.pil_to_tensor(image).to(torch.float32) / 255.0
    image_tensor = TF.normalize(image_tensor, IMAGE_MEAN, IMAGE_STD)

    trimap_tensor = TF.pil_to_tensor(trimap)
    binary = trimap_to_binary(trimap_tensor)
    box = mask_to_box(binary)
    return image_tensor, PetTarget(box_xyxy=box, mask=binary)


class OxfordPetTasks(Dataset[tuple[Tensor, PetTarget]]):
    """Task wrapper exposing one binary mask and one derived pet box."""

    def __init__(
        self,
        root: Path,
        split: str,
        download: bool,
        train: bool,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.train = train
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, PetTarget]:
        image, trimap = self.dataset[index]
        flip = self.train and random.random() < 0.5
        return prepare_sample(
            image,
            trimap,
            image_size=self.image_size,
            horizontal_flip=flip,
        )


def deterministic_split(
    dataset: Dataset[tuple[Tensor, PetTarget]],
    val_fraction: float,
    seed: int,
) -> tuple[Dataset[tuple[Tensor, PetTarget]], Dataset[tuple[Tensor, PetTarget]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def limit_dataset(
    dataset: Dataset[tuple[Tensor, PetTarget]],
    maximum: int | None,
) -> Dataset[tuple[Tensor, PetTarget]]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def collate(
    batch: list[tuple[Tensor, PetTarget]],
) -> tuple[Tensor, Tensor, Tensor]:
    images = torch.stack([sample[0] for sample in batch])
    boxes = torch.stack([sample[1].box_xyxy for sample in batch])
    masks = torch.stack([sample[1].mask for sample in batch])
    return images, boxes, masks


def dataset_metadata() -> dict[str, Any]:
    return {
        "name": "Oxford-IIIT Pet",
        "task": "pet detection + binary segmentation",
        "image_size": IMAGE_SIZE,
        "mask_policy": "labels 1 and 3 foreground; label 2 background",
        "license": "CC BY-SA 4.0",
    }
