#!/usr/bin/env python3
"""Module 34: grounded VQA with deterministic patch-question attention."""
from __future__ import annotations

import argparse
import math


def softmax(logits: list[float]) -> list[float]:
    if not logits or not all(math.isfinite(v) for v in logits):
        raise ValueError("logits must be nonempty and finite")
    maximum = max(logits)
    exp_values = [math.exp(value - maximum) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def ground(
    question: list[float], patches: list[list[float]], labels: list[str]
) -> tuple[str, list[float]]:
    """Select the patch most relevant to a question; return label and attention."""
    if not question or not all(math.isfinite(x) for x in question):
        raise ValueError("invalid question embedding")
    if not patches or len(patches) != len(labels):
        raise ValueError("patches and labels must be nonempty and aligned")
    if any(len(patch) != len(question) or not all(math.isfinite(x) for x in patch)
           for patch in patches):
        raise ValueError("invalid patch embedding")
    if any(not label for label in labels):
        raise ValueError("labels must be nonempty")
    scale = math.sqrt(len(question))
    logits = [sum(a * b for a, b in zip(question, patch, strict=True)) / scale
              for patch in patches]
    attention = softmax(logits)
    best = max(range(len(attention)), key=attention.__getitem__)
    return labels[best], attention


def synthetic() -> tuple[str, list[float]]:
    return ground([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], ["red car", "blue bus"])


def real_vqa(image_path: str, question: str, checkpoint: str) -> str:
    """Optional pretrained VQA pipeline; downloads weights on first use."""
    from PIL import Image
    from transformers import pipeline

    model = pipeline("visual-question-answering", model=checkpoint, device=-1)
    with Image.open(image_path) as image:
        answers = model(image=image.convert("RGB"), question=question)
    if not answers or "answer" not in answers[0]:
        raise RuntimeError("model returned no answer")
    return str(answers[0]["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--question")
    parser.add_argument("--checkpoint", default="dandelin/vilt-b32-finetuned-vqa")
    args = parser.parse_args()
    if args.synthetic:
        answer, attention = synthetic()
        print(answer)
        for value in attention:
            print(f"{value:.9f}")
    elif args.image and args.question:
        print(real_vqa(args.image, args.question, args.checkpoint))
    else:
        parser.error("provide --synthetic or both --image and --question")


if __name__ == "__main__":
    main()
