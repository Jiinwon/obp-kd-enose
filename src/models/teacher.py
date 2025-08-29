"""Teacher model that fuses time-series features with docking priors."""
from __future__ import annotations

import torch
from torch import nn

from .backbones import SimpleCNN


class TeacherModel(nn.Module):
    """Teacher model that fuses time‑series features with docking priors.

    Two fusion mechanisms are supported:

    * ``use_mlp=False`` (default): late bilinear fusion ``s_c = zᵀ W p_c``
    * ``use_mlp=True``: simple MLP on the concatenation ``[z; p_c]``

    In addition to the classification head a small regression head predicting
    TVOC (total volatile organic compound) concentration is provided.  For
    backwards compatibility the forward method returns only the classification
    logits by default but the regression output can be requested via
    ``return_tvoc=True``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        prior_dim: int = 4,
        hidden_dim: int = 32,
        *,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = SimpleCNN(in_channels=1, out_dim=hidden_dim)
        self.prior_fc = nn.Linear(prior_dim, hidden_dim)
        self.use_mlp = use_mlp

        if use_mlp:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
        else:
            # Bilinear weight matrix W used in z^T W p
            self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.regressor = nn.Linear(hidden_dim, 1)

    def _fuse(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.use_mlp:
            return self.fusion(torch.cat([z, p], dim=1))
        # Bilinear late fusion: (z W) ⊙ p
        return torch.matmul(z, self.weight) * p

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        *,
        return_tvoc: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute logits (and optionally TVOC estimate).

        Parameters
        ----------
        x:
            Input time‑series of shape ``(B, C, L)``.
        prior:
            Prior vectors of shape ``(B, prior_dim)``.
        return_tvoc:
            When ``True`` also return the regression output.
        """

        z = self.backbone(x)
        p = torch.relu(self.prior_fc(prior))
        feat = self._fuse(z, p)
        logits = self.classifier(feat)
        tvoc = self.regressor(feat).squeeze(-1)
        if return_tvoc:
            return logits, tvoc
        return logits


__all__ = ["TeacherModel"]
