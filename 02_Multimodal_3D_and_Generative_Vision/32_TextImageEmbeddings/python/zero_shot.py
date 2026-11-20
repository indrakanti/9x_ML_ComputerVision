#!/usr/bin/env python3
"""Module 32: deterministic embedding parity and optional real CLIP inference."""
import argparse
import math


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm <= 1e-12:
        raise ValueError("zero-norm embedding")
    return [float(v) / norm for v in values]


def scores(image: list[float], texts: list[list[float]], temperature: float = 1.0) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    im = normalize(image)
    result = []
    for text in texts:
        if len(text) != len(im):
            raise ValueError("embedding dimension mismatch")
        tx = normalize(text)
        result.append(sum(a * b for a, b in zip(im, tx, strict=True)) / temperature)
    return result


def real_example(image_path: str, labels: list[str]) -> None:
    from PIL import Image
    import torch
    from transformers import CLIPModel, CLIPProcessor

    checkpoint = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(checkpoint)
    model = CLIPModel.from_pretrained(checkpoint).eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=list(labels), images=image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
        probabilities = logits.softmax(dim=-1)
    for label, probability in sorted(zip(labels, probabilities.tolist(), strict=True), key=lambda x: -x[1]):
        print(f"{label}: {probability:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--labels", nargs="+", default=["a dog", "a cat", "a car"])
    args = parser.parse_args()
    if args.synthetic:
        for value in scores([3, 4], [[3, 4], [4, -3], [-3, -4]]):
            print(f"{value:.6f}")
    elif args.image:
        real_example(args.image, args.labels)
    else:
        parser.error("provide --synthetic or --image")


if __name__ == "__main__":
    main()
