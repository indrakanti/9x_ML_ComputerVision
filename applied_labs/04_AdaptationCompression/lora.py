"""LoRA layers and injection helpers for Applied Lab 04."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Linear layer with frozen base weights and trainable low-rank update."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias: bool,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__(input_features, output_features, bias=bias)
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nn.Parameter(
            torch.empty(rank, input_features)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(output_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        base: nn.Linear,
        rank: int,
        alpha: float,
    ) -> LoRALinear:
        layer = cls(
            base.in_features,
            base.out_features,
            base.bias is not None,
            rank,
            alpha,
        )
        with torch.no_grad():
            layer.weight.copy_(base.weight)
            if base.bias is not None and layer.bias is not None:
                layer.bias.copy_(base.bias)
        layer.weight.requires_grad_(False)
        if layer.bias is not None:
            layer.bias.requires_grad_(False)
        return layer

    def forward(self, input_tensor: Tensor) -> Tensor:
        base = F.linear(input_tensor, self.weight, self.bias)
        low_rank = F.linear(input_tensor, self.lora_a)
        update = F.linear(low_rank, self.lora_b)
        return base + (self.alpha / self.rank) * update


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
) -> int:
    """Replace Linear children recursively and return number replaced."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear.from_linear(child, rank, alpha),
            )
            replaced += 1
        else:
            replaced += inject_lora(child, rank, alpha)
    return replaced


def freeze_except_lora(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(
            name.endswith("lora_a") or name.endswith("lora_b")
        )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def total_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
