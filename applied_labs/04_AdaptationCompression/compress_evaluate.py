"""Evaluate pruning and dynamic INT8 quantization on a trained Lab-01 model."""

from __future__ import annotations

import argparse
import copy
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from data import cifar, limited
from lab01_models import build_model, checkpoint_model_name
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/applied_lab_04/compression.json"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prune-amount", type=float, default=0.30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def loader(
    dataset: Dataset[Any],
    batch_size: int,
    workers: int,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def accuracy(model: nn.Module, data_loader: DataLoader[Any]) -> float:
    model.eval()
    correct = 0
    samples = 0
    for images, labels in data_loader:
        logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += labels.shape[0]
    return correct / samples


def apply_unstructured_pruning(
    model: nn.Module,
    amount: float,
) -> tuple[nn.Module, float]:
    if not 0.0 <= amount < 1.0:
        raise ValueError("prune amount must be in [0, 1)")

    pruned = copy.deepcopy(model).cpu()
    prunable: list[nn.Module] = []
    for module in pruned.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prunable.append(module)

    zeros = 0
    total = 0
    for module in prunable:
        weight = module.weight
        zeros += int((weight == 0).sum().item())
        total += weight.numel()
        prune.remove(module, "weight")

    return pruned, zeros / max(total, 1)


def dynamic_int8(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def serialized_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


@torch.inference_mode()
def latency_report(
    model: nn.Module,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    model.eval()
    sample = torch.randn(1, 3, 32, 32)

    for _ in range(warmup):
        _ = model(sample)

    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model(sample)
        values.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(values)
    p95_index = min(
        len(ordered) - 1,
        int(round((len(ordered) - 1) * 0.95)),
    )
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
    }


def evaluate_variant(
    model: nn.Module,
    data_loader: DataLoader[Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    return {
        "accuracy": accuracy(model, data_loader),
        "state_bytes": serialized_size_bytes(model),
        "latency": latency_report(model, warmup, iterations),
    }


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model_name = checkpoint_model_name(checkpoint)
    baseline = build_model(model_name).cpu()
    baseline.load_state_dict(checkpoint["model_state_dict"])
    baseline.eval()

    test: Dataset[Any] = cifar(
        args.data_dir,
        train=False,
        download=args.download,
        domain="standard",
        augment=False,
    )
    test = limited(test, args.max_test_samples)
    test_loader = loader(
        test,
        args.batch_size,
        args.num_workers,
    )

    pruned, achieved_sparsity = apply_unstructured_pruning(
        baseline,
        args.prune_amount,
    )
    quantized = dynamic_int8(baseline)

    report = {
        "lab": "04_AdaptationCompression",
        "experiment": "pruning_and_dynamic_int8",
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "test_samples": len(test),
        "prune_amount": args.prune_amount,
        "achieved_weight_sparsity": achieved_sparsity,
        "baseline": evaluate_variant(
            baseline,
            test_loader,
            args.warmup,
            args.iterations,
        ),
        "pruned": evaluate_variant(
            pruned,
            test_loader,
            args.warmup,
            args.iterations,
        ),
        "dynamic_int8": evaluate_variant(
            quantized,
            test_loader,
            args.warmup,
            args.iterations,
        ),
        "quantization_note": (
            "dynamic INT8 targets supported Linear operators; "
            "unsupported operators remain floating point"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
