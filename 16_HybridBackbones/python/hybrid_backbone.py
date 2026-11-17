"""Hybrid CNN-transformer reference for Module 16."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class HybridArtifact:
    image_h: int
    image_w: int
    kernel: int
    stride: int
    embed: int
    conv_weight: tuple[float, ...]
    conv_bias: float
    proj_weight: tuple[float, ...]
    proj_bias: tuple[float, ...]
    wq: tuple[float, ...]
    wk: tuple[float, ...]
    wv: tuple[float, ...]


def _vec(v: dict[str, str], key: str) -> tuple[float, ...]:
    return tuple(float(x) for x in v[key].split(","))


def load_artifact(path: Path) -> HybridArtifact:
    v: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        k, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"invalid line: {raw}")
        v[k] = value

    if int(v["version"]) != 1:
        raise ValueError("unsupported version")
    return HybridArtifact(
        image_h=int(v["image_h"]),
        image_w=int(v["image_w"]),
        kernel=int(v["kernel"]),
        stride=int(v["stride"]),
        embed=int(v["embed"]),
        conv_weight=_vec(v, "conv_weight"),
        conv_bias=float(v["conv_bias"]),
        proj_weight=_vec(v, "proj_weight"),
        proj_bias=_vec(v, "proj_bias"),
        wq=_vec(v, "wq"),
        wk=_vec(v, "wk"),
        wv=_vec(v, "wv"),
    )


def manual_forward(a: HybridArtifact, image: list[float]) -> list[float]:
    out_h = (a.image_h - a.kernel) // a.stride + 1
    out_w = (a.image_w - a.kernel) // a.stride + 1
    conv: list[float] = []
    for y in range(out_h):
        for x in range(out_w):
            value = a.conv_bias
            for ky in range(a.kernel):
                for kx in range(a.kernel):
                    iy = y * a.stride + ky
                    ix = x * a.stride + kx
                    value += (
                        image[iy * a.image_w + ix]
                        * a.conv_weight[ky * a.kernel + kx]
                    )
            conv.append(value)

    tokens = [
        [
            a.proj_bias[d] + a.proj_weight[d] * value
            for d in range(a.embed)
        ]
        for value in conv
    ]

    def project(rows: list[list[float]], w: tuple[float, ...]) -> list[list[float]]:
        return [
            [
                sum(w[o * a.embed + i] * row[i] for i in range(a.embed))
                for o in range(a.embed)
            ]
            for row in rows
        ]

    q, k, v = project(tokens, a.wq), project(tokens, a.wk), project(tokens, a.wv)
    scale = math.sqrt(float(a.embed))
    output: list[float] = []
    for query in q:
        scores = [
            sum(query[d] * key[d] for d in range(a.embed)) / scale for key in k
        ]
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        total = sum(ex)
        p = [z / total for z in ex]
        for d in range(a.embed):
            output.append(sum(p[i] * v[i][d] for i in range(len(v))))
    return output


def pytorch_forward(a: HybridArtifact, image: list[float]) -> Tensor:
    x = torch.tensor(image, dtype=torch.float64).reshape(1, 1, a.image_h, a.image_w)
    conv_w = torch.tensor(a.conv_weight, dtype=torch.float64).reshape(
        1, 1, a.kernel, a.kernel
    )
    conv = torch.nn.functional.conv2d(
        x,
        conv_w,
        torch.tensor([a.conv_bias], dtype=torch.float64),
        stride=a.stride,
    )
    values = conv.reshape(-1, 1)
    proj_w = torch.tensor(a.proj_weight, dtype=torch.float64).reshape(a.embed, 1)
    tokens = values @ proj_w.T + torch.tensor(a.proj_bias, dtype=torch.float64)

    def matrix(values_: tuple[float, ...]) -> Tensor:
        return torch.tensor(values_, dtype=torch.float64).reshape(a.embed, a.embed)

    q = tokens @ matrix(a.wq).T
    k = tokens @ matrix(a.wk).T
    v = tokens @ matrix(a.wv).T
    attn = torch.softmax(q @ k.T / math.sqrt(float(a.embed)), dim=-1)
    return attn @ v


def example_image() -> list[float]:
    return [float(i) for i in range(1, 17)]
