"""Simple backbone models."""
from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Very small 1D CNN encoder."""

    def __init__(self, in_channels: int = 1, out_dim: int = 8) -> None:
        super().__init__()
        # TODO: replace with a proper CNN encoder for time-series data.
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` of shape ``(B, C, L)`` into ``(B, out_dim)``."""
        # TODO: extend with deeper network and normalisation layers.
        x = self.net(x)
        return x.view(x.size(0), -1)


class SimpleTCN(nn.Module):
    """Placeholder temporal convolution network."""

    def __init__(self, in_channels: int = 1, out_dim: int = 8) -> None:
        super().__init__()
        # TODO: implement real TCN with dilated convolutions.
        self.conv = nn.Conv1d(in_channels, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` of shape ``(B, C, L)`` into ``(B, out_dim)``."""
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)
