"""Trainable CIFAR-10 backbones for Applied Lab 01."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


class TinyCNN(nn.Module):
    """Small CNN suitable for CPU training on CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class TinyViT(nn.Module):
    """Compact ViT for 32x32 CIFAR images."""

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")

        grid = image_size // patch_size
        tokens = grid * grid
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, tokens + 1, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        batch = x.shape[0]
        cls = self.class_token.expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.position
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Build one Applied-Lab-01 backbone without downloading weights."""
    if name == "tinycnn":
        return TinyCNN(num_classes=num_classes)

    if name == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        return model

    if name == "tinyvit":
        return TinyViT(num_classes=num_classes)

    raise ValueError(f"unsupported model: {name}")


def parameter_count(model: nn.Module) -> int:
    """Return total model parameter count."""
    return sum(parameter.numel() for parameter in model.parameters())
