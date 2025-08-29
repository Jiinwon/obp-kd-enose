"""Teacher model that fuses time-series features with docking priors."""
from __future__ import annotations

import torch
from torch import nn

from .backbones import SimpleCNN


class TeacherModel(nn.Module):
    """Simple fusion model.

    Parameters
    ----------
    num_classes: int
        Number of output classes.
    prior_dim: int
        Dimension of prior vector.
    """

    def __init__(self, num_classes: int = 3, prior_dim: int = 4) -> None:
        super().__init__()
        self.backbone = SimpleCNN()
        self.prior_fc = nn.Linear(prior_dim, 8)
        self.classifier = nn.Linear(self.backbone.out_dim + 8, num_classes)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        ts_feat = self.backbone(x)
        prior_feat = torch.relu(self.prior_fc(prior))
        feat = torch.cat([ts_feat, prior_feat], dim=1)
        return self.classifier(feat)
