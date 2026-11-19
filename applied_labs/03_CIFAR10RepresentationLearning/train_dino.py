"""Train a compact DINO-style student/teacher model on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from data import TwoViewCIFAR
from models import DINOBackbone, ema_update, make_teacher
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from utils import (
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
        default=Path("runs/applied_lab_03/dino"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
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


def dino_loss(
    student_a: Tensor,
    student_b: Tensor,
    teacher_a: Tensor,
    teacher_b: Tensor,
    center: Tensor,
    student_temperature: float,
    teacher_temperature: float,
) -> Tensor:
    student_a_log = F.log_softmax(
        student_a / student_temperature,
        dim=1,
    )
    student_b_log = F.log_softmax(
        student_b / student_temperature,
        dim=1,
    )
    teacher_a_prob = F.softmax(
        (teacher_a - center) / teacher_temperature,
        dim=1,
    ).detach()
    teacher_b_prob = F.softmax(
        (teacher_b - center) / teacher_temperature,
        dim=1,
    ).detach()

    a_to_b = -(teacher_b_prob * student_a_log).sum(dim=1).mean()
    b_to_a = -(teacher_a_prob * student_b_log).sum(dim=1).mean()
    return 0.5 * (a_to_b + b_to_a)


@torch.no_grad()
def update_center(
    center: Tensor,
    teacher_a: Tensor,
    teacher_b: Tensor,
    momentum: float,
) -> None:
    batch_center = torch.cat((teacher_a, teacher_b), dim=0).mean(
        dim=0,
        keepdim=True,
    )
    center.mul_(momentum).add_(
        batch_center,
        alpha=1.0 - momentum,
    )


def run_epoch(
    student: DINOBackbone,
    teacher: DINOBackbone,
    loader: Any,
    center: Tensor,
    device: torch.device,
    student_temperature: float,
    teacher_temperature: float,
    teacher_momentum: float,
    center_momentum: float,
    optimizer: AdamW | None,
) -> float:
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    total = 0.0
    samples = 0

    for view_a, view_b, _ in loader:
        view_a = view_a.to(device, non_blocking=True)
        view_b = view_b.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_a = teacher(view_a)
            teacher_b = teacher(view_b)

        with torch.set_grad_enabled(training):
            student_a = student(view_a)
            student_b = student(view_b)
            loss = dino_loss(
                student_a,
                student_b,
                teacher_a,
                teacher_b,
                center,
                student_temperature,
                teacher_temperature,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
                ema_update(teacher, student, teacher_momentum)
                update_center(
                    center,
                    teacher_a,
                    teacher_b,
                    center_momentum,
                )

        batch = view_a.shape[0]
        total += float(loss.detach().item()) * batch
        samples += batch

    return total / samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train: Dataset[Any] = TwoViewCIFAR(
        args.data_dir,
        train=True,
        download=args.download,
    )
    val: Dataset[Any] = TwoViewCIFAR(
        args.data_dir,
        train=False,
        download=args.download,
    )
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

    student = DINOBackbone().to(device)
    teacher = make_teacher(student).to(device)
    optimizer = AdamW(student.parameters(), lr=args.learning_rate)
    center = torch.zeros(1, 128, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            student,
            teacher,
            train_loader,
            center,
            device,
            args.student_temperature,
            args.teacher_temperature,
            args.teacher_momentum,
            args.center_momentum,
            optimizer,
        )
        val_loss = run_epoch(
            student,
            teacher,
            val_loader,
            center,
            device,
            args.student_temperature,
            args.teacher_temperature,
            args.teacher_momentum,
            args.center_momentum,
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
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "center": center.detach().cpu(),
            "epoch": epoch,
            "seed": args.seed,
            "student_temperature": args.student_temperature,
            "teacher_temperature": args.teacher_temperature,
            "teacher_momentum": args.teacher_momentum,
            "center_momentum": args.center_momentum,
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
        "experiment": "dino_style",
        "device": str(device),
        "epochs": args.epochs,
        "train_samples": len(train),
        "val_samples": len(val),
        "best_val_loss": best_val_loss,
        "student_temperature": args.student_temperature,
        "teacher_temperature": args.teacher_temperature,
        "teacher_momentum": args.teacher_momentum,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
