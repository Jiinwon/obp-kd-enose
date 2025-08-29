"""Optional alignment losses (placeholders)."""
from __future__ import annotations

import torch


def cosine_align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity based alignment."""
    a_n = a / a.norm(dim=1, keepdim=True)
    b_n = b / b.norm(dim=1, keepdim=True)
    return 1 - (a_n * b_n).sum(dim=1).mean()
