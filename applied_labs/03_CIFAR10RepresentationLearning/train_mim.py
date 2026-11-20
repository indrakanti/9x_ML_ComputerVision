"""Train a tiny masked-image model on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from models import TinyMIM
from torch.optim import AdamW
from torch.utils.data import Dataset, Subset
from utils import (
    cifar_dataset,
    limit_dataset,
    make_loader,
    resolve_device,
    set_seed,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_03/mim"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def random_mask(
    batch: int,
    tokens: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 < ratio < 1.0:
        raise ValueError("mask ratio must be in (0, 1)")
    mask_count = max(1, int(round(tokens * ratio)))
    noise = torch.rand(batch, tokens, device=device)
    indices = noise.argsort(dim=1)
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, indices[:, :mask_count], True)
    return mask


def run_epoch(
    model: TinyMIM,
    loader: Any,
    device: torch.device,
    mask_ratio: float,
    optimizer: AdamW | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        mask = random_mask(
            images.shape[0],
            model.tokens,
            mask_ratio,
            device,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            _, loss = model(images, mask)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch = images.shape[0]
        total += float(loss.detach().item()) * batch
        samples += batch

    return total / samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_augmented = cifar_dataset(
        args.data_dir,
        train=True,
        download=args.download,
        augment=True,
    )
    train_eval = cifar_dataset(
        args.data_dir,
        train=True,
        download=False,
        augment=False,
    )
    train_indices, val_indices = split_indices(
        len(train_augmented),
        args.val_fraction,
        args.seed,
    )
    train: Dataset[Any] = Subset(train_augmented, train_indices)
    val: Dataset[Any] = Subset(train_eval, val_indices)
    train = limit_dataset(train, args.max_train_samples)
    val = limit_dataset(val, args.max_val_samples)

    train_loader = make_loader(
        train,
        args.batch_size,
        args.num_workers,
        True,
        device,
    )
    val_loader = make_loader(
        val,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )

    model = TinyMIM().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            args.mask_ratio,
            optimizer,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device,
            args.mask_ratio,
            None,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "mask_ratio": args.mask_ratio,
            "best_val_loss": min(best_val_loss, val_loss),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, args.output_dir / "best.pt")

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f}"
        )

    report = {
        "lab": "03_CIFAR10RepresentationLearning",
        "experiment": "masked_image_modeling",
        "device": str(device),
        "epochs": args.epochs,
        "mask_ratio": args.mask_ratio,
        "train_samples": len(train),
        "val_samples": len(val),
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
