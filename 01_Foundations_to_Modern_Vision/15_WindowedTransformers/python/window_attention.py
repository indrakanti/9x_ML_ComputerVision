"""Windowed attention reference for Module 15."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class WindowArtifact:
    grid_h: int
    grid_w: int
    embed: int
    window: int
    wq: tuple[float, ...]
    wk: tuple[float, ...]
    wv: tuple[float, ...]


def _vec(values: dict[str, str], key: str) -> tuple[float, ...]:
    return tuple(float(x) for x in values[key].split(","))


def load_artifact(path: Path) -> WindowArtifact:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"invalid artifact line: {raw}")
        values[key] = value

    artifact = WindowArtifact(
        grid_h=int(values["grid_h"]),
        grid_w=int(values["grid_w"]),
        embed=int(values["embed"]),
        window=int(values["window"]),
        wq=_vec(values, "wq"),
        wk=_vec(values, "wk"),
        wv=_vec(values, "wv"),
    )
    if int(values["version"]) != 1:
        raise ValueError("unsupported artifact version")
    if artifact.grid_h % artifact.window or artifact.grid_w % artifact.window:
        raise ValueError("grid must be divisible by window")
    expected = artifact.embed * artifact.embed
    if any(len(m) != expected for m in (artifact.wq, artifact.wk, artifact.wv)):
        raise ValueError("attention matrix size mismatch")
    return artifact


def partition_windows(
    artifact: WindowArtifact,
    tokens: list[float],
) -> list[list[list[float]]]:
    expected = artifact.grid_h * artifact.grid_w * artifact.embed
    if len(tokens) != expected:
        raise ValueError("token count mismatch")

    windows: list[list[list[float]]] = []
    for top in range(0, artifact.grid_h, artifact.window):
        for left in range(0, artifact.grid_w, artifact.window):
            window_tokens: list[list[float]] = []
            for y in range(artifact.window):
                for x in range(artifact.window):
                    token_index = (top + y) * artifact.grid_w + left + x
                    start = token_index * artifact.embed
                    window_tokens.append(tokens[start : start + artifact.embed])
            windows.append(window_tokens)
    return windows


def _project(
    rows: list[list[float]],
    weights: tuple[float, ...],
    dim: int,
) -> list[list[float]]:
    return [
        [
            sum(weights[o * dim + i] * row[i] for i in range(dim))
            for o in range(dim)
        ]
        for row in rows
    ]


def manual_window_attention(
    artifact: WindowArtifact,
    tokens: list[float],
) -> list[float]:
    output: list[float] = []
    scale = math.sqrt(float(artifact.embed))

    for window in partition_windows(artifact, tokens):
        q = _project(window, artifact.wq, artifact.embed)
        k = _project(window, artifact.wk, artifact.embed)
        v = _project(window, artifact.wv, artifact.embed)

        for query in q:
            scores = [
                sum(query[d] * key[d] for d in range(artifact.embed)) / scale
                for key in k
            ]
            maximum = max(scores)
            exp_values = [math.exp(score - maximum) for score in scores]
            total = sum(exp_values)
            probabilities = [value / total for value in exp_values]
            for dim in range(artifact.embed):
                output.append(
                    sum(
                        probabilities[index] * v[index][dim]
                        for index in range(len(v))
                    )
                )
    return output


def pytorch_window_attention(
    artifact: WindowArtifact,
    tokens: list[float],
) -> Tensor:
    windows = torch.tensor(
        partition_windows(artifact, tokens),
        dtype=torch.float64,
    )
    wq = torch.tensor(artifact.wq, dtype=torch.float64).reshape(
        artifact.embed, artifact.embed
    )
    wk = torch.tensor(artifact.wk, dtype=torch.float64).reshape(
        artifact.embed, artifact.embed
    )
    wv = torch.tensor(artifact.wv, dtype=torch.float64).reshape(
        artifact.embed, artifact.embed
    )

    q = windows @ wq.T
    k = windows @ wk.T
    v = windows @ wv.T
    scores = q @ k.transpose(-2, -1) / math.sqrt(float(artifact.embed))
    attention = torch.softmax(scores, dim=-1)
    return attention @ v


def example_tokens() -> list[float]:
    return [float(i) / 10.0 for i in range(32)]
