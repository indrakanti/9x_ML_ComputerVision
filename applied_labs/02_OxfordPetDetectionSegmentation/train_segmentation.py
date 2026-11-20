"""Train TinyUNet on Oxford-IIIT Pet trimaps."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data import OxfordPetTasks, collate, dataset_metadata, limit_dataset
from models import TinyUNet
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford-pet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_02/segmentation"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


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


def split_indices(
    length: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_count = max(1, int(round(length * val_fraction)))
    return indices[val_count:], indices[:val_count]


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
        collate_fn=collate,
    )


def soft_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
    ) + soft_dice_loss(logits, target)


def train_epoch(
    model: TinyUNet,
    loader: DataLoader[Any],
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    samples = 0

    for images, _, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()

        batch = images.shape[0]
        total_loss += float(loss.detach().item()) * batch
        samples += batch

    return total_loss / samples


@torch.inference_mode()
def evaluate(
    model: TinyUNet,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    intersection = 0.0
    union = 0.0
    prediction_area = 0.0
    target_area = 0.0

    for images, _, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        predicted = torch.sigmoid(model(images)) >= 0.5
        target = masks >= 0.5

        intersection += float((predicted & target).sum().item())
        union += float((predicted | target).sum().item())
        prediction_area += float(predicted.sum().item())
        target_area += float(target.sum().item())

    iou = intersection / max(union, 1.0)
    dice = (
        2.0 * intersection
        / max(prediction_area + target_area, 1.0)
    )
    return {"iou": iou, "dice": dice}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_source = OxfordPetTasks(
        args.data_dir,
        "trainval",
        args.download,
        train=True,
    )
    eval_source = OxfordPetTasks(
        args.data_dir,
        "trainval",
        False,
        train=False,
    )
    test_source = OxfordPetTasks(
        args.data_dir,
        "test",
        args.download,
        train=False,
    )
    train_indices, val_indices = split_indices(
        len(train_source),
        args.val_fraction,
        args.seed,
    )

    train_dataset: Dataset[Any] = Subset(train_source, train_indices)
    val_dataset: Dataset[Any] = Subset(eval_source, val_indices)
    test_dataset: Dataset[Any] = test_source
    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    val_dataset = limit_dataset(val_dataset, args.max_val_samples)
    test_dataset = limit_dataset(test_dataset, args.max_test_samples)

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        True,
        device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )

    model = TinyUNet().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        seconds = time.perf_counter() - started

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_iou": val["iou"],
                "val_dice": val["dice"],
                "seconds": seconds,
            }
        )
        checkpoint = {
            "model_name": "tinyunet",
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "dataset": dataset_metadata(),
            "best_val_iou": max(best_val_iou, val["iou"]),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val["iou"] > best_val_iou:
            best_val_iou = val["iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")

        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"val_iou={val['iou']:.4f} "
            f"val_dice={val['dice']:.4f} seconds={seconds:.1f}"
        )

    best = torch.load(
        args.output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, test_loader, device)

    metrics = {
        "lab": "02_OxfordPetDetectionSegmentation",
        "task": "binary_segmentation",
        "model": "tinyunet",
        "dataset": dataset_metadata(),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_iou": best_val_iou,
        "test_iou": test["iou"],
        "test_dice": test["dice"],
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
