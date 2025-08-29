"""Lightweight student model."""
from __future__ import annotations

import torch
from torch import nn

from .backbones import SimpleCNN


class StudentModel(nn.Module):
    """Time-series only classifier."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = SimpleCNN()
        self.classifier = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)
