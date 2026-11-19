"""Benchmark an Applied-Lab-01 checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from models import build_model, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        int(round((len(ordered) - 1) * fraction)),
    )
    return ordered[index]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model_name = str(checkpoint["model_name"])
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = torch.randn(args.batch_size, 3, 32, 32, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(sample)
        synchronize(device)

        latency_ms: list[float] = []
        for _ in range(args.iterations):
            synchronize(device)
            start = time.perf_counter()
            _ = model(sample)
            synchronize(device)
            latency_ms.append((time.perf_counter() - start) * 1000.0)

    report = {
        "model": model_name,
        "device": str(device),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "parameters": parameter_count(model),
        "mean_ms": statistics.fmean(latency_ms),
        "median_ms": statistics.median(latency_ms),
        "p95_ms": percentile(latency_ms, 0.95),
        "min_ms": min(latency_ms),
        "max_ms": max(latency_ms),
    }

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
