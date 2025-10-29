"""Temporal convolutional network with FiLM modulation."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class FiLMBlock(nn.Module):
    """Residual 1D convolution block with FiLM modulation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        x = self.activation(x)
        return x + residual


class FiLMTCNStudent(nn.Module):
    """Temporal network that modulates activations with docking priors."""

    def __init__(
        self,
        in_channels: int,
        prior_dim: int,
        hidden_dim: int,
        n_blocks: int,
        kernel_size: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden_dim, kernel_size=kernel_size, dilation=2 ** i) for i in range(n_blocks)]
        )
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prior_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(hidden_dim, n_classes)
        self.prior_projection = nn.Linear(hidden_dim, prior_dim)

    def forward(self, x: Tensor, prior: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass returning logits and projected prior."""

        if prior is None:
            raise ValueError("FiLMTCNStudent requires a prior vector for modulation")
        prior = nn.functional.normalize(prior, dim=-1)
        x = self.input_proj(x)
        for block, generator in zip(self.blocks, self.film_generators):
            gains_bias = generator(prior)
            gamma, beta = gains_bias.chunk(2, dim=-1)
            x = block(x, gamma, beta)
        pooled = x.mean(dim=-1)
        logits = self.head(pooled)
        z_hat = self.prior_projection(pooled)
        return logits, z_hat


__all__ = ["FiLMTCNStudent"]
