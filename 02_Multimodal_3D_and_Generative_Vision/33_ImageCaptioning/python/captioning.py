#!/usr/bin/env python3
"""Module 33: deterministic encoder/decoder attention and optional real captioning."""
from __future__ import annotations

import argparse
import math


def softmax(logits: list[float]) -> list[float]:
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("logits must be nonempty and finite")
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def attend(query: list[float], patches: list[list[float]]) -> tuple[list[float], list[float]]:
    """Scaled dot-product cross attention for one decoder query and image patch tokens."""
    if not query or not patches or not all(math.isfinite(x) for x in query):
        raise ValueError("query and patches must be nonempty and finite")
    if any(len(patch) != len(query) or not all(math.isfinite(x) for x in patch)
           for patch in patches):
        raise ValueError("patch dimensions or values invalid")
    scale = math.sqrt(len(query))
    weights = softmax([sum(a * b for a, b in zip(query, patch, strict=True)) / scale
                       for patch in patches])
    context = [sum(weight * patch[i] for weight, patch in zip(weights, patches, strict=True))
               for i in range(len(query))]
    return weights, context


def synthetic() -> list[float]:
    weights, context = attend([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    return weights + context


def real_caption(image_path: str, checkpoint: str) -> str:
    """Optional pretrained image-to-text encoder/decoder; requires model download."""
    from PIL import Image
    from transformers import pipeline

    captioner = pipeline("image-to-text", model=checkpoint, device=-1)
    with Image.open(image_path) as image:
        result = captioner(image.convert("RGB"))
    if not result or "generated_text" not in result[0]:
        raise RuntimeError("caption model returned no text")
    return str(result[0]["generated_text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image", help="Path to an image for optional pretrained inference")
    parser.add_argument("--checkpoint", default="Salesforce/blip-image-captioning-base")
    args = parser.parse_args()
    if args.synthetic:
        for value in synthetic():
            print(f"{value:.9f}")
    elif args.image:
        print(real_caption(args.image, args.checkpoint))
    else:
        parser.error("specify --synthetic or --image")


if __name__ == "__main__":
    main()
