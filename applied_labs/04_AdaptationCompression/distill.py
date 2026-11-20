"""Distill a trained Lab-01 teacher into TinyCNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from data import cifar, limited
from lab01_models import build_model, checkpoint_model_name
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_04/distillation"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
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


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float,
    alpha: float,
) -> Tensor:
    hard = F.cross_entropy(student_logits, labels)
    student_log = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    soft = F.kl_div(
        student_log,
        teacher_prob,
        reduction="batchmean",
    ) * temperature * temperature
    return alpha * soft + (1.0 - alpha) * hard


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
def evaluate(
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
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be > 0")

    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.teacher_checkpoint,
        map_location=device,
        weights_only=False,
    )
    teacher_name = checkpoint_model_name(checkpoint)
    teacher = build_model(teacher_name).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = build_model("tinycnn").to(device)

    train: Dataset[Any] = cifar(
        args.data_dir,
        train=True,
        download=args.download,
        domain="standard",
        augment=True,
    )
    test: Dataset[Any] = cifar(
        args.data_dir,
        train=False,
        download=args.download,
        domain="standard",
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

    optimizer = AdamW(student.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    teacher_accuracy = evaluate(teacher, test_loader, device)

    for epoch in range(1, args.epochs + 1):
        student.train()
        loss_sum = 0.0
        samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits = teacher(images)

            optimizer.zero_grad(set_to_none=True)
            student_logits = student(images)
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                args.temperature,
                args.alpha,
            )
            loss.backward()
            optimizer.step()

            batch = labels.shape[0]
            loss_sum += float(loss.detach().item()) * batch
            samples += batch

        test_accuracy = evaluate(student, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / samples,
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"epoch={epoch:03d} loss={loss_sum / samples:.4f} "
            f"student_test_acc={test_accuracy:.4f}"
        )

    torch.save(
        {
            "model_name": "tinycnn",
            "model_state_dict": student.state_dict(),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "temperature": args.temperature,
            "alpha": args.alpha,
        },
        args.output_dir / "student.pt",
    )
    report = {
        "lab": "04_AdaptationCompression",
        "experiment": "distillation",
        "teacher_model": teacher_name,
        "teacher_test_accuracy": teacher_accuracy,
        "student_model": "tinycnn",
        "temperature": args.temperature,
        "alpha": args.alpha,
        "final_student_test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
