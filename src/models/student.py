"""Lightweight student model."""
from __future__ import annotations

import torch
from torch import nn

from .backbones import SimpleCNN


class StudentModel(nn.Module):
    """Light‑weight time‑series model used for deployment."""

    def __init__(self, num_classes: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.backbone = SimpleCNN(in_channels=1, out_dim=hidden_dim)
        self.classifier = nn.Linear(self.backbone.out_dim, num_classes)
        self.regressor = nn.Linear(self.backbone.out_dim, 1)

    def forward(
        self, x: torch.Tensor, *, return_tvoc: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute logits (and optionally a TVOC estimate)."""

        feat = self.backbone(x)
        logits = self.classifier(feat)
        tvoc = self.regressor(feat).squeeze(-1)
        if return_tvoc:
            return logits, tvoc
        return logits


__all__ = ["StudentModel"]
