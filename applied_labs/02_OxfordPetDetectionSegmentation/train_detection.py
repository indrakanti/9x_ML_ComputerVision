"""Train TinyYOLO or TinyDETR on Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data import OxfordPetTasks, collate, dataset_metadata, limit_dataset
from geometry import box_iou_aligned, cxcywh_to_xyxy, xyxy_to_cxcywh
from models import TinyDETR, TinyYOLO
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yolo", "detr"), default="yolo")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford-pet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/applied_lab_02/detection"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(
    length: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_count = max(1, int(round(length * val_fraction)))
    return indices[val_count:], indices[:val_count]


def make_loader(
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
        collate_fn=collate,
    )


def yolo_targets(
    boxes_xyxy: Tensor,
    grid_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    boxes = xyxy_to_cxcywh(boxes_xyxy)
    cx, cy, width, height = boxes.unbind(dim=1)

    grid_x = torch.clamp((cx * grid_size).long(), max=grid_size - 1)
    grid_y = torch.clamp((cy * grid_size).long(), max=grid_size - 1)
    offsets = torch.stack(
        (
            cx * grid_size - grid_x,
            cy * grid_size - grid_y,
            width,
            height,
        ),
        dim=1,
    )

    objectness = torch.zeros(
        boxes.shape[0],
        grid_size,
        grid_size,
        device=boxes.device,
    )
    batch_indices = torch.arange(boxes.shape[0], device=boxes.device)
    objectness[batch_indices, grid_y, grid_x] = 1.0
    return objectness, offsets, grid_x, grid_y


def yolo_loss(model: TinyYOLO, images: Tensor, boxes_xyxy: Tensor) -> Tensor:
    output = model(images)
    object_target, box_target, grid_x, grid_y = yolo_targets(
        boxes_xyxy,
        model.grid_size,
    )
    object_logits = output[:, 0]
    pos_weight = torch.tensor(5.0, device=images.device)
    object_loss = F.binary_cross_entropy_with_logits(
        object_logits,
        object_target,
        pos_weight=pos_weight,
    )

    batch_indices = torch.arange(images.shape[0], device=images.device)
    selected = output[
        batch_indices,
        1:5,
        grid_y,
        grid_x,
    ]
    box_prediction = torch.sigmoid(selected)
    box_loss = F.smooth_l1_loss(box_prediction, box_target)
    return object_loss + 5.0 * box_loss


def decode_yolo(model: TinyYOLO, images: Tensor) -> Tensor:
    output = model(images)
    batch = output.shape[0]
    objectness = torch.sigmoid(output[:, 0])
    flat_index = objectness.flatten(1).argmax(dim=1)
    grid_y = flat_index // model.grid_size
    grid_x = flat_index % model.grid_size
    batch_indices = torch.arange(batch, device=images.device)
    selected = torch.sigmoid(
        output[
            batch_indices,
            1:5,
            grid_y,
            grid_x,
        ]
    )
    cx = (grid_x.to(torch.float32) + selected[:, 0]) / model.grid_size
    cy = (grid_y.to(torch.float32) + selected[:, 1]) / model.grid_size
    boxes = torch.stack(
        (cx, cy, selected[:, 2], selected[:, 3]),
        dim=1,
    )
    return cxcywh_to_xyxy(boxes)


def detr_loss(model: TinyDETR, images: Tensor, boxes_xyxy: Tensor) -> Tensor:
    class_logits, pred_cxcywh = model(images)
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)
    targets_cxcywh = xyxy_to_cxcywh(boxes_xyxy)

    pet_probability = class_logits.softmax(dim=-1)[..., 1]
    l1 = torch.abs(
        pred_cxcywh - targets_cxcywh[:, None, :]
    ).sum(dim=-1)
    expanded_targets = boxes_xyxy[:, None, :].expand_as(pred_xyxy)
    iou = box_iou_aligned(pred_xyxy, expanded_targets)
    matching_cost = -pet_probability + 5.0 * l1 + 2.0 * (1.0 - iou)
    matched_query = matching_cost.argmin(dim=1)

    class_target = torch.zeros(
        class_logits.shape[:2],
        dtype=torch.long,
        device=images.device,
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    class_target[batch_indices, matched_query] = 1

    classification_loss = F.cross_entropy(
        class_logits.reshape(-1, 2),
        class_target.reshape(-1),
    )
    matched_boxes = pred_cxcywh[batch_indices, matched_query]
    matched_xyxy = pred_xyxy[batch_indices, matched_query]
    box_loss = F.l1_loss(matched_boxes, targets_cxcywh)
    iou_loss = (
        1.0 - box_iou_aligned(matched_xyxy, boxes_xyxy)
    ).mean()
    return classification_loss + 5.0 * box_loss + 2.0 * iou_loss


def decode_detr(model: TinyDETR, images: Tensor) -> Tensor:
    class_logits, boxes_cxcywh = model(images)
    pet_probability = class_logits.softmax(dim=-1)[..., 1]
    best_query = pet_probability.argmax(dim=1)
    batch_indices = torch.arange(images.shape[0], device=images.device)
    return cxcywh_to_xyxy(boxes_cxcywh[batch_indices, best_query])


def model_loss(
    model: nn.Module,
    model_name: str,
    images: Tensor,
    boxes: Tensor,
) -> Tensor:
    if model_name == "yolo":
        if not isinstance(model, TinyYOLO):
            raise TypeError("YOLO model type mismatch")
        return yolo_loss(model, images, boxes)

    if not isinstance(model, TinyDETR):
        raise TypeError("DETR model type mismatch")
    return detr_loss(model, images, boxes)


def decode_boxes(
    model: nn.Module,
    model_name: str,
    images: Tensor,
) -> Tensor:
    if model_name == "yolo":
        if not isinstance(model, TinyYOLO):
            raise TypeError("YOLO model type mismatch")
        return decode_yolo(model, images)

    if not isinstance(model, TinyDETR):
        raise TypeError("DETR model type mismatch")
    return decode_detr(model, images)


def train_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader[Any],
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    samples = 0

    for images, boxes, _ in loader:
        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model, model_name, images, boxes)
        loss.backward()
        optimizer.step()

        batch = images.shape[0]
        total_loss += float(loss.detach().item()) * batch
        samples += batch

    return total_loss / samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ious: list[Tensor] = []

    for images, boxes, _ in loader:
        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        predicted = decode_boxes(model, model_name, images)
        ious.append(box_iou_aligned(predicted, boxes).cpu())

    all_iou = torch.cat(ious)
    return {
        "mean_iou": float(all_iou.mean().item()),
        "recall_at_0_5": float((all_iou >= 0.5).to(torch.float32).mean().item()),
    }


def build_model(name: str) -> nn.Module:
    if name == "yolo":
        return TinyYOLO()
    if name == "detr":
        return TinyDETR()
    raise ValueError(f"unsupported model: {name}")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_source = OxfordPetTasks(
        args.data_dir,
        "trainval",
        args.download,
        train=True,
    )
    eval_source = OxfordPetTasks(
        args.data_dir,
        "trainval",
        False,
        train=False,
    )
    test_source = OxfordPetTasks(
        args.data_dir,
        "test",
        args.download,
        train=False,
    )
    train_indices, val_indices = split_indices(
        len(train_source),
        args.val_fraction,
        args.seed,
    )
    train_dataset: Dataset[Any] = Subset(train_source, train_indices)
    val_dataset: Dataset[Any] = Subset(eval_source, val_indices)
    test_dataset: Dataset[Any] = test_source

    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    val_dataset = limit_dataset(val_dataset, args.max_val_samples)
    test_dataset = limit_dataset(test_dataset, args.max_test_samples)

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        True,
        device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )

    model = build_model(args.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_mean_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        loss = train_epoch(
            model,
            args.model,
            train_loader,
            optimizer,
            device,
        )
        val = evaluate(model, args.model, val_loader, device)
        seconds = time.perf_counter() - started

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": loss,
            "val_mean_iou": val["mean_iou"],
            "val_recall_at_0_5": val["recall_at_0_5"],
            "seconds": seconds,
        }
        history.append(row)

        checkpoint = {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "dataset": dataset_metadata(),
            "best_val_mean_iou": max(best_mean_iou, val["mean_iou"]),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val["mean_iou"] > best_mean_iou:
            best_mean_iou = val["mean_iou"]
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"epoch={epoch:03d} loss={loss:.4f} "
            f"val_iou={val['mean_iou']:.4f} "
            f"val_r50={val['recall_at_0_5']:.4f} "
            f"seconds={seconds:.1f}"
        )

    best = torch.load(
        run_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, args.model, test_loader, device)

    metrics = {
        "lab": "02_OxfordPetDetectionSegmentation",
        "task": "detection",
        "model": args.model,
        "dataset": dataset_metadata(),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_mean_iou": best_mean_iou,
        "test_mean_iou": test["mean_iou"],
        "test_recall_at_0_5": test["recall_at_0_5"],
        "history": history,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
