"""Optional alignment losses (placeholders)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity based alignment loss.

    Returns ``1 - cos(a, b)`` averaged over the batch.
    """

    a_n = F.normalize(a, dim=1)
    b_n = F.normalize(b, dim=1)
    return 1 - (a_n * b_n).sum(dim=1).mean()


def info_nce(z: torch.Tensor, p: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE loss between feature vectors ``z`` and class priors ``p``.

    Both ``z`` and ``p`` are normalised and all pairwise similarities are
    computed.  The diagonal of the resulting similarity matrix is treated as the
    positive pairs.
    """

    z_n = F.normalize(z, dim=1)
    p_n = F.normalize(p, dim=1)
    logits = torch.matmul(z_n, p_n.t()) / temperature
    labels = torch.arange(z_n.size(0), device=z_n.device)
    return F.cross_entropy(logits, labels)


__all__ = ["cosine_align", "info_nce"]
