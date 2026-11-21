"""Evaluate frozen MIM or DINO encoders with a linear CIFAR-10 probe."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from models import DINOBackbone, TinyMIM
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from utils import (
    cifar_dataset,
    limit_dataset,
    make_loader,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("mim", "dino"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/applied_lab_03/linear_probe.json"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def load_encoder(
    kind: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, Callable[[Tensor], Tensor], int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if kind == "mim":
        model = TinyMIM().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        def extract(images: Tensor) -> Tensor:
            return model.representation(images)

        dimension = 128
    else:
        model = DINOBackbone().to(device)
        model.load_state_dict(checkpoint["student_state_dict"])

        def extract(images: Tensor) -> Tensor:
            return model.representation(images)

        dimension = 128

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, extract, dimension


def evaluate(
    extract: Callable[[Tensor], Tensor],
    classifier: nn.Linear,
    loader: Any,
    device: torch.device,
) -> float:
    classifier.eval()
    correct = 0
    samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = classifier(extract(images))
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            samples += labels.shape[0]

    return correct / samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train: Dataset[Any] = cifar_dataset(
        args.data_dir,
        train=True,
        download=args.download,
        augment=True,
    )
    test: Dataset[Any] = cifar_dataset(
        args.data_dir,
        train=False,
        download=args.download,
        augment=False,
    )
    train = limit_dataset(train, args.max_train_samples)
    test = limit_dataset(test, args.max_test_samples)

    train_loader = make_loader(
        train,
        args.batch_size,
        args.num_workers,
        True,
        device,
    )
    test_loader = make_loader(
        test,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )

    _, extract, feature_dim = load_encoder(
        args.encoder,
        args.checkpoint,
        device,
    )
    classifier = nn.Linear(feature_dim, 10).to(device)
    optimizer = AdamW(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                features = extract(images)

            optimizer.zero_grad(set_to_none=True)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch = labels.shape[0]
            total_loss += float(loss.detach().item()) * batch
            samples += batch

        test_accuracy = evaluate(
            extract,
            classifier,
            test_loader,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / samples,
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={total_loss / samples:.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )

    report = {
        "lab": "03_CIFAR10RepresentationLearning",
        "experiment": "linear_probe",
        "encoder": args.encoder,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "epochs": args.epochs,
        "train_samples": len(train),
        "test_samples": len(test),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
