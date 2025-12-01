"""Feed-forward classifiers for the distilled pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


__all__ = [
    "MLP",
    "build_mlp",
    "TeacherConfig",
    "StudentConfig",
    "build_teacher",
    "build_student",
]


class MLP(nn.Module):
    """Simple multi-layer perceptron classifier."""

    def __init__(self, layers: Sequence[int], *, dropout: float | None = None) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("At least input and output layer sizes must be provided")
        modules: list[nn.Module] = []
        last_index = len(layers) - 1
        for idx, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(in_dim, out_dim))
            if idx != last_index - 1:
                modules.append(nn.ReLU())
                if dropout and dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(x)


def build_mlp(input_dim: int, hidden_layers: Iterable[int], num_classes: int) -> MLP:
    """Factory for :class:`MLP` with convenience arguments."""

    layers = [int(input_dim), *[int(h) for h in hidden_layers], int(num_classes)]
    return MLP(layers)


@dataclass(frozen=True)
class TeacherConfig:
    input_dim: int = 128
    hidden: Sequence[int] = (256, 128)
    num_classes: int = 6


@dataclass(frozen=True)
class StudentConfig:
    input_dim: int = 128
    hidden: Sequence[int] = (128, 64)
    num_classes: int = 6


def build_teacher(cfg: TeacherConfig) -> MLP:
    """Instantiate the teacher MLP."""

    return build_mlp(cfg.input_dim, cfg.hidden, cfg.num_classes)


def build_student(cfg: StudentConfig) -> MLP:
    """Instantiate the student MLP."""

    return build_mlp(cfg.input_dim, cfg.hidden, cfg.num_classes)

