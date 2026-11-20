"""Small representation-learning models for Applied Lab 03."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.projection(x)


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embed_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embed_dim,
            padding_idx=0,
        )
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        valid = (tokens != 0).unsqueeze(-1)
        summed = (embedded * valid).sum(dim=1)
        count = valid.sum(dim=1).clamp(min=1)
        return self.projection(summed / count)


class TinyCLIP(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embed_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(
            vocabulary_size,
            output_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(1.0 / temperature))))
        )

    def forward(
        self,
        images: Tensor,
        prompt_tokens: Tensor,
    ) -> Tensor:
        image_embeddings = F.normalize(
            self.image_encoder(images),
            dim=1,
        )
        text_embeddings = F.normalize(
            self.text_encoder(prompt_tokens),
            dim=1,
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * image_embeddings @ text_embeddings.T


class TinyMIM(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.grid = 32 // patch_size
        self.tokens = self.grid * self.grid
        self.patch_projection = nn.Linear(self.patch_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(
            torch.zeros(1, self.tokens, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.decoder = nn.Linear(embed_dim, self.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def patchify(self, images: Tensor) -> Tensor:
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patches.transpose(1, 2)

    def encode_tokens(
        self,
        images: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        patches = self.patchify(images)
        tokens = self.patch_projection(patches)
        if mask is not None:
            mask_token = self.mask_token.expand(
                images.shape[0],
                self.tokens,
                -1,
            )
            tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)
        return self.encoder(tokens + self.position)

    def representation(self, images: Tensor) -> Tensor:
        return self.encode_tokens(images).mean(dim=1)

    def forward(
        self,
        images: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target = self.patchify(images)
        encoded = self.encode_tokens(images, mask)
        reconstruction = self.decoder(encoded)
        squared = (reconstruction - target) ** 2
        masked_loss = squared[mask].mean()
        return reconstruction, masked_loss


class DINOBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ImageEncoder(feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.projector(self.encoder(images))

    def representation(self, images: Tensor) -> Tensor:
        return self.encoder(images)


def make_teacher(student: DINOBackbone) -> DINOBackbone:
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    momentum: float,
) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(),
        student.parameters(),
        strict=True,
    ):
        teacher_parameter.mul_(momentum).add_(
            student_parameter,
            alpha=1.0 - momentum,
        )
