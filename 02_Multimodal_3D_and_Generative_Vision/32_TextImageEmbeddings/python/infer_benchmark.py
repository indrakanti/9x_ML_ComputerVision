#!/usr/bin/env python3
"""Run production Module32 image classification and measure ONNX CPU inference."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from production import CLASSES, PROMPTS, encode_text


def load_image(path: Path) -> np.ndarray:
    """Exactly match CIFAR-10 training: RGB, 32x32, float32 NCHW / 255."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (32, 32):
            rgb = rgb.resize((32, 32), Image.Resampling.BILINEAR)
        return (np.asarray(rgb, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]


def infer(
    directory: Path, image: np.ndarray, warmup: int = 5, repeats: int = 30
) -> dict[str, object]:
    if warmup < 0 or repeats < 1 or image.shape != (1, 3, 32, 32):
        raise ValueError("invalid image or benchmark settings")
    image_model = ort.InferenceSession(
        str(directory / "image.onnx"), providers=["CPUExecutionProvider"]
    )
    text_model = ort.InferenceSession(
        str(directory / "text.onnx"), providers=["CPUExecutionProvider"]
    )
    tokens = np.asarray([encode_text(prompt) for prompt in PROMPTS], dtype=np.int64)
    image_name = image_model.get_inputs()[0].name
    text_name = text_model.get_inputs()[0].name
    text_embeddings = text_model.run(None, {text_name: tokens})[0]
    for _ in range(warmup):
        image_model.run(None, {image_name: image})
    durations = []
    image_embedding = np.empty((1, 32), dtype=np.float32)
    for _ in range(repeats):
        start = time.perf_counter_ns()
        image_embedding = image_model.run(None, {image_name: image})[0]
        durations.append((time.perf_counter_ns() - start) / 1e6)
    scores = (10.0 * image_embedding @ text_embeddings.T)[0]
    scores -= scores.max()
    probabilities = np.exp(scores) / np.exp(scores).sum()
    ranking = np.argsort(-probabilities)
    latency = np.asarray(durations, dtype=np.float64)
    return {
        "class": CLASSES[int(ranking[0])],
        "ranked": [
            {"label": CLASSES[int(index)], "probability": float(probabilities[index])}
            for index in ranking
        ],
        "benchmark": {
            "backend": "onnxruntime_cpu",
            "scope": "warm image encoder only; text embeddings calculated once",
            "warmup": warmup,
            "repeats": repeats,
            "latency_p50_ms": float(np.percentile(latency, 50)),
            "latency_p95_ms": float(np.percentile(latency, 95)),
            "latency_mean_ms": float(latency.mean()),
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(
        infer(args.export_dir, load_image(args.image), args.warmup, args.repeats),
        indent=2,
    ))


if __name__ == "__main__":
    main()
