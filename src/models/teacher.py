"""Teacher model that fuses time-series features with docking priors."""
from __future__ import annotations

import torch
from torch import nn

from .backbones import SimpleCNN


class TeacherModel(nn.Module):
    """Minimal bilinear teacher network.

    TODO: replace with full architecture that performs knowledge distillation
    and additional regression heads.
    """

    def __init__(
        self,
        num_classes: int = 3,
        prior_dim: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        # TODO: swap SimpleCNN for a stronger encoder.
        self.backbone = SimpleCNN(in_channels=1, out_dim=hidden_dim)
        # Bilinear projection matrix W:[H,D]
        self.weight = nn.Parameter(torch.randn(hidden_dim, prior_dim))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities.

        Parameters
        ----------
        x: torch.Tensor
            Sensor sequence of shape ``(B, 1, L)``.
        priors: torch.Tensor
            Prior matrix ``P`` of shape ``(C, D)`` for ``C`` classes.

        Returns
        -------
        torch.Tensor
            Class probabilities ``(B, C)``.
        """

        # z: [B, H]
        z = self.backbone(x)
        # scores: [B, C] via bilinear form z @ W @ P.T
        scores = z @ self.weight @ priors.T
        probs = scores.softmax(dim=1)
        return probs


__all__ = ["TeacherModel"]
