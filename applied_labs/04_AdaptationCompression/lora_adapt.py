"""LoRA-adapt a trained TinyViT checkpoint to grayscale CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from data import cifar, limited
from lab01_models import build_model, checkpoint_model_name
from lora import (
    freeze_except_lora,
    inject_lora,
    total_parameter_count,
    trainable_parameter_count,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_04/lora"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
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


def loader(
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


@torch.inference_mode()
def accuracy(
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    samples = 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += labels.shape[0]
    return correct / samples


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model_name = checkpoint_model_name(checkpoint)
    if model_name != "tinyvit":
        raise ValueError("LoRA lab expects a TinyViT checkpoint from Lab 01")

    train: Dataset[Any] = cifar(
        args.data_dir,
        train=True,
        download=args.download,
        domain="grayscale",
        augment=True,
    )
    test: Dataset[Any] = cifar(
        args.data_dir,
        train=False,
        download=args.download,
        domain="grayscale",
        augment=False,
    )
    train = limited(train, args.max_train_samples)
    test = limited(test, args.max_test_samples)

    train_loader = loader(
        train,
        args.batch_size,
        args.num_workers,
        True,
        device,
    )
    test_loader = loader(
        test,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    baseline_accuracy = accuracy(model, test_loader, device)

    replaced = inject_lora(model, args.rank, args.alpha)
    if replaced == 0:
        raise RuntimeError("no Linear layers were replaced by LoRA")
    freeze_except_lora(model)

    trainable = trainable_parameter_count(model)
    total = total_parameter_count(model)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch = labels.shape[0]
            loss_sum += float(loss.detach().item()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            samples += batch

        test_accuracy = accuracy(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / samples,
                "train_accuracy": correct / samples,
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"epoch={epoch:03d} loss={loss_sum / samples:.4f} "
            f"train_acc={correct / samples:.4f} "
            f"test_acc={test_accuracy:.4f}"
        )

    report = {
        "lab": "04_AdaptationCompression",
        "experiment": "lora_grayscale_adaptation",
        "checkpoint": str(args.checkpoint),
        "rank": args.rank,
        "alpha": args.alpha,
        "replaced_linear_layers": replaced,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "baseline_grayscale_accuracy": baseline_accuracy,
        "adapted_grayscale_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "rank": args.rank,
            "alpha": args.alpha,
            "base_checkpoint": str(args.checkpoint),
        },
        args.output_dir / "adapted.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
