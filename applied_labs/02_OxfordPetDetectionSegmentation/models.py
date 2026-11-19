"""Trainable detector and segmentation models for Applied Lab 02."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvEncoder(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TinyYOLO(nn.Module):
    """One-anchor-per-cell YOLO-style detector."""

    def __init__(self, grid_size: int = 4) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.encoder = ConvEncoder(128)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.head = nn.Conv2d(128, 5, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.pool(self.encoder(x)))


class TinyDETR(nn.Module):
    """Small DETR-style detector with learned object queries."""

    def __init__(
        self,
        num_queries: int = 5,
        hidden_dim: int = 128,
        heads: int = 4,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.class_head = nn.Linear(hidden_dim, 2)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(x)
        memory = features.flatten(2).transpose(1, 2)
        batch = x.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        class_logits = self.class_head(decoded)
        boxes_cxcywh = torch.sigmoid(self.box_head(decoded))
        return class_logits, boxes_cxcywh


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.bottleneck = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        center = self.bottleneck(self.pool(enc2))

        dec2 = self.up2(center)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        return self.head(dec1).squeeze(1)
