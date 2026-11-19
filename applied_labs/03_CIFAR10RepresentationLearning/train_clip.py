"""Train a small CLIP-style class-prompt model on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from data import character_vocabulary, encode_prompts
from models import TinyCLIP
from torch import nn
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
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_03/clip"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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


def run_epoch(
    model: TinyCLIP,
    loader: Any,
    prompt_tokens: torch.Tensor,
    device: torch.device,
    optimizer: AdamW | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images, prompt_tokens)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += batch

    return total_loss / samples, correct / samples


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

    prompts = encode_prompts().to(device)
    vocabulary_size = max(character_vocabulary().values()) + 1
    model = TinyCLIP(vocabulary_size=vocabulary_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            prompts,
            device,
            optimizer,
        )
        test_loss, test_accuracy = run_epoch(
            model,
            test_loader,
            prompts,
            device,
            None,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "vocabulary": character_vocabulary(),
            "prompt_tokens": prompts.cpu(),
            "best_test_accuracy": max(best_accuracy, test_accuracy),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(checkpoint, args.output_dir / "best.pt")

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"test_acc={test_accuracy:.4f}"
        )

    report: dict[str, Any] = {
        "lab": "03_CIFAR10RepresentationLearning",
        "experiment": "clip_style",
        "device": str(device),
        "epochs": args.epochs,
        "train_samples": len(train),
        "test_samples": len(test),
        "best_test_accuracy": best_accuracy,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
