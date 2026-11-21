#!/usr/bin/env python3
"""Module 35: visual token projection, fusion and optional pretrained VLM."""
from __future__ import annotations

import argparse
import math


def project(patches: list[list[float]], weights: list[list[float]]) -> list[list[float]]:
    """Map vision encoder patch tokens into the language-model embedding width."""
    if not patches or not weights or not weights[0]:
        raise ValueError("empty patches or projection")
    width = len(weights)
    output = len(weights[0])
    if any(len(row) != output or not all(math.isfinite(x) for x in row) for row in weights):
        raise ValueError("invalid projection")
    if any(len(patch) != width or not all(math.isfinite(x) for x in patch) for patch in patches):
        raise ValueError("patch width mismatch or nonfinite values")
    return [
        [sum(patch[k] * weights[k][j] for k in range(width)) for j in range(output)]
        for patch in patches
    ]


def fuse(
    visual: list[list[float]], text: list[list[float]], image_token_position: int
) -> list[list[float]]:
    """Insert projected visual tokens at the image placeholder in a text sequence."""
    if not visual or not text or not 0 <= image_token_position < len(text):
        raise ValueError("invalid fusion inputs")
    width = len(text[0])
    if width == 0 or any(len(token) != width for token in visual + text):
        raise ValueError("embedding width mismatch")
    if any(not all(math.isfinite(x) for x in token) for token in visual + text):
        raise ValueError("nonfinite token")
    return text[:image_token_position] + visual + text[image_token_position + 1:]


def synthetic() -> list[list[float]]:
    patches = [[1.0, 2.0], [3.0, 4.0]]
    weights = [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]
    text = [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0], [8.0, 8.0, 8.0]]
    return fuse(project(patches, weights), text, 1)


def real_example(image_path: str, prompt: str, checkpoint: str) -> str:
    """Optional real VLM inference; requires compatible transformers and model weights."""
    import torch
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("image-text-to-text", model=checkpoint, device=-1, dtype=torch.float32)
    with Image.open(image_path) as image:
        results = pipe(images=image.convert("RGB"), text=prompt, max_new_tokens=64)
    if not results:
        raise RuntimeError("VLM returned no output")
    return str(results[0].get("generated_text", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--checkpoint", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    args = parser.parse_args()
    if args.synthetic:
        for token in synthetic():
            print(" ".join(f"{value:.6f}" for value in token))
    elif args.image:
        print(real_example(args.image, args.prompt, args.checkpoint))
    else:
        parser.error("provide --synthetic or --image")


if __name__ == "__main__":
    main()
