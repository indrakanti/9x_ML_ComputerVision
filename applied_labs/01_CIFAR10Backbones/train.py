"""Train CIFAR-10 backbones for Applied Lab 01."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import build_model, parameter_count
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("tinycnn", "resnet18", "tinyvit"),
        default="tinycnn",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_01"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited(
    dataset: Dataset[tuple[Tensor, int]],
    maximum: int | None,
) -> Dataset[tuple[Tensor, int]]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def build_datasets(
    data_dir: Path,
    download: bool,
    seed: int,
) -> tuple[Dataset[tuple[Tensor, int]], Dataset[tuple[Tensor, int]], Dataset[tuple[Tensor, int]]]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_augmented = CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=download,
    )
    train_evaluation = CIFAR10(
        root=data_dir,
        train=True,
        transform=eval_transform,
        download=False,
    )
    test = CIFAR10(
        root=data_dir,
        train=False,
        transform=eval_transform,
        download=download,
    )

    generator = torch.Generator().manual_seed(seed)
    train_size = 45_000
    val_size = len(train_augmented) - train_size
    train_indices, val_indices = random_split(
        range(len(train_augmented)),
        [train_size, val_size],
        generator=generator,
    )

    train = Subset(train_augmented, list(train_indices))
    val = Subset(train_evaluation, list(val_indices))
    return train, val, test


def make_loader(
    dataset: Dataset[tuple[Tensor, int]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch = labels.shape[0]
        loss_sum += float(loss.detach().item()) * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch

    return loss_sum / total, correct / total


def save_curve(history: list[EpochMetrics], path: Path) -> None:
    epochs = [row.epoch for row in history]
    train_accuracy = [row.train_accuracy for row in history]
    val_accuracy = [row.val_accuracy for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_accuracy, label="train accuracy")
    plt.plot(epochs, val_accuracy, label="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset, val_dataset, test_dataset = build_datasets(
        args.data_dir,
        args.download,
        args.seed,
    )
    train_dataset = limited(train_dataset, args.max_train_samples)
    val_dataset = limited(val_dataset, args.max_val_samples)
    test_dataset = limited(test_dataset, args.max_test_samples)

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

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[EpochMetrics] = []
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
        )
        seconds = time.perf_counter() - started
        scheduler.step()

        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            seconds=seconds,
        )
        history.append(row)

        checkpoint = {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "best_val_accuracy": max(best_val_accuracy, val_accuracy),
            "dataset": "CIFAR-10",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"seconds={seconds:.1f}"
        )

    best = torch.load(
        run_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    test_loss, test_accuracy = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        None,
    )

    metrics = {
        "lab": "01_CIFAR10Backbones",
        "dataset": "CIFAR-10",
        "model": args.model,
        "parameters": parameter_count(model),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": [asdict(row) for row in history],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_curve(history, run_dir / "learning_curve.png")

    print(
        f"test_loss={test_loss:.4f} "
        f"test_accuracy={test_accuracy:.4f} "
        f"parameters={parameter_count(model)}"
    )
    print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
