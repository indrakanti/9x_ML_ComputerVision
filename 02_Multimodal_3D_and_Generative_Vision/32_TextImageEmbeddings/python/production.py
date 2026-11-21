#!/usr/bin/env python3
"""Train, evaluate, export and verify a small reusable image/text dual encoder.

CIFAR-10 labels are the supervised training task; arbitrary prompts are not
claimed to generalize like pretrained CLIP. No dataset or model is bundled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
PROMPTS = tuple(f"a photo of a {label}" for label in CLASSES)
MAX_TOKENS = 32
EMBED_DIM = 32


def encode_text(prompt: str) -> list[int]:
    """Fixed ASCII tokenizer; non-ASCII characters are rejected, not silently changed."""
    if not prompt or len(prompt) > MAX_TOKENS or any(ord(c) < 32 or ord(c) > 126 for c in prompt):
        raise ValueError("prompt must be 1..32 printable ASCII characters")
    return [ord(c) - 31 for c in prompt] + [0] * (MAX_TOKENS - len(prompt))


def prompt_tokens(device: torch.device | str = "cpu") -> Tensor:
    return torch.tensor([encode_text(text) for text in PROMPTS], dtype=torch.long, device=device)


class ImageTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, EMBED_DIM),
        )

    def forward(self, images: Tensor) -> Tensor:
        return F.normalize(self.features(images), dim=-1)


class TextTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(96, EMBED_DIM, padding_idx=0)
        self.projection = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embedding(tokens)
        mask = (tokens != 0).unsqueeze(-1)
        mean = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.projection(mean), dim=-1)


class DualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = ImageTower()
        self.text = TextTower()

    def forward(self, images: Tensor, tokens: Tensor) -> Tensor:
        return 10.0 * (self.image(images) @ self.text(tokens).T)


class SyntheticImages(Dataset[tuple[Tensor, int]]):
    """Deterministic labeled patterns for smoke tests, not benchmark results."""

    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        label = index % len(CLASSES)
        image = torch.zeros(3, 32, 32)
        image[label % 3].fill_(0.1 + 0.08 * label)
        image[:, label * 3:label * 3 + 2, :].fill_(0.7)
        return image, label


def make_dataset(name: str, root: Path, train: bool, download: bool) -> Dataset[tuple[Tensor, int]]:
    if name == "synthetic":
        return SyntheticImages(100 if train else 40)
    if name != "cifar10":
        raise ValueError("dataset must be cifar10 or synthetic")
    from torchvision import datasets, transforms

    return datasets.CIFAR10(
        root=str(root), train=train, download=download, transform=transforms.ToTensor()
    )


def subset(\n    dataset: Dataset[tuple[Tensor, int]], limit: int, seed: int\n) -> Dataset[tuple[Tensor, int]]:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    indices = np.random.default_rng(seed).permutation(len(dataset))[:limit].tolist()
    return Subset(dataset, indices)


def evaluate(model: DualEncoder, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    total = 0
    correct = 0
    top5 = 0
    per_class = [[0, 0] for _ in CLASSES]
    tokens = prompt_tokens(device)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device), tokens)
            labels = labels.to(device)
            prediction = logits.argmax(dim=1)
            correct += int((prediction == labels).sum())
            top5 += int((logits.topk(5, dim=1).indices == labels[:, None]).any(dim=1).sum())
            total += int(labels.numel())
            for cls in range(len(CLASSES)):
                selected = labels == cls
                per_class[cls][1] += int(selected.sum())
                per_class[cls][0] += int(((prediction == labels) & selected).sum())
    if not total:
        raise ValueError("empty evaluation set")
    return {
        "dataset_samples": total,
        "top1": correct / total,
        "top5": top5 / total,
        "chance_top1": 1.0 / len(CLASSES),
        "per_class_top1": {
            label: (hits / count if count else None)
            for label, (hits, count) in zip(CLASSES, per_class, strict=True)
        },
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    training = subset(\n        make_dataset(args.dataset, args.data_root, True, args.download),\n        args.max_train, args.seed,\n    )
    testing = subset(\n        make_dataset(args.dataset, args.data_root, False, args.download),\n        args.max_test, args.seed + 1,\n    )
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False)
    model = DualEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    texts = prompt_tokens(device)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels in train_loader:
            logits = model(images.to(device), texts)
            loss = F.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": float(epoch + 1), "loss": sum(losses) / len(losses)})
    metrics = evaluate(model, test_loader, device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": list(CLASSES), "seed": args.seed,
         "preprocessing": "RGB float32 CHW / 255, 32x32; ASCII fixed 32 tokens"},
        args.output / "model.pt",
    )
    report = {"dataset": args.dataset, "seed": args.seed, "train_samples": len(training),
              "test_samples": len(testing), "epochs": args.epochs, "history": history,
              "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def load_model(checkpoint: Path) -> DualEncoder:
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data["classes"] != list(CLASSES):
        raise ValueError("checkpoint label order mismatch")
    model = DualEncoder()
    model.load_state_dict(data["state_dict"])
    return model.eval()


def export(checkpoint: Path, output: Path) -> None:
    import onnx
    import onnxruntime as ort

    model = load_model(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "image.onnx"
    text_path = output / "text.onnx"
    images = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    tokens = prompt_tokens()
    for tower, example, name, dest in [
        (model.image, images, "images", image_path),
        (model.text, tokens, "tokens", text_path),
    ]:
        torch.onnx.export(
            tower, example, str(dest), input_names=[name], output_names=["embeddings"],
            dynamic_axes={name: {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(dest)))
        native = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        actual = native.run(None, {name: example.numpy()})[0]
        expected = tower(example).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.savetxt(output / "image_input.txt", images.numpy().ravel(), fmt="%.9g")
    np.savetxt(output / "text_input.txt", tokens.numpy().ravel(), fmt="%d")
    manifest = {
        "format": "onnx", "opset": 17, "image_shape": [1, 3, 32, 32],
        "text_shape": [10, MAX_TOKENS], "embedding_width": EMBED_DIM,
        "labels": list(CLASSES), "prompts": list(PROMPTS),
        "image_preprocessing": "RGB; resize to 32x32; float32 CHW in [0,1]",
        "tokenizer": "printable ASCII codepoint minus 31; pad 0 to 32",
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (image_path, text_path)
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", choices=("synthetic", "cifar10"), default="cifar10")
    training.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    training.add_argument("--download", action="store_true")
    training.add_argument("--output", type=Path, default=Path("runs/module32"))
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--max-train", type=int, default=0)
    training.add_argument("--max-test", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    exporting = commands.add_parser("export")
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("epochs and batch size must be positive")
        train(args)
    else:
        export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
