"""Student model supporting FiLM-TCN backbone."""
from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from .backbones import SimpleCNN
from .film_tcn import FiLMTCNStudent


class StudentModel(nn.Module):
    """Flexible student model with optional FiLM modulation."""

    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 32,
        in_channels: int = 1,
        *,
        use_film: bool = False,
        prior_dim: Optional[int] = None,
        film_hidden: int = 64,
        film_blocks: int = 3,
        film_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.use_film = use_film
        if use_film:
            if prior_dim is None:
                raise ValueError("prior_dim must be provided when use_film=True")
            self.backbone = FiLMTCNStudent(
                in_channels=in_channels,
                prior_dim=prior_dim,
                hidden_dim=film_hidden,
                n_blocks=film_blocks,
                kernel_size=film_kernel,
                n_classes=num_classes,
            )
            self.classifier = None
            self.regressor = None
            self.prior_dim = prior_dim
        else:
            self.backbone = SimpleCNN(in_channels=in_channels, out_dim=hidden_dim)
            self.classifier = nn.Linear(self.backbone.out_dim, num_classes)
            self.regressor = nn.Linear(self.backbone.out_dim, 1)
            self.prior_dim = None

    def forward(
        self,
        x: Tensor,
        *,
        prior: Optional[Tensor] = None,
        return_tvoc: bool = False,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, L)``.
        prior:
            Docking prior vectors used when ``use_film`` is enabled.
        return_tvoc:
            Backward-compatible flag returning a TVOC estimate for the legacy CNN backbone.
        return_aux:
            When ``use_film`` is enabled, return the projected prior ``z_hat``.
        """

        if self.use_film:
            logits, z_hat = self.backbone(x, prior=prior)
            if return_aux:
                return logits, z_hat
            return logits
        feat = self.backbone(x)
        logits = self.classifier(feat)
        tvoc = self.regressor(feat).squeeze(-1)
        if return_tvoc or return_aux:
            return logits, tvoc
        return logits


__all__ = ["StudentModel"]
