"""Knowledge distillation losses."""
from __future__ import annotations

import torch
from torch import nn


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute the KL-divergence based KD loss."""
    log_p = nn.functional.log_softmax(student_logits / temperature, dim=1)
    q = nn.functional.softmax(teacher_logits / temperature, dim=1)
    return nn.functional.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)
