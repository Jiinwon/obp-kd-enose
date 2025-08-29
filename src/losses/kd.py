"""Knowledge distillation losses."""
from __future__ import annotations

import torch
from torch import nn


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the standard KL‑divergence based KD loss.

    This function is kept for backwards compatibility with earlier exercises.
    Prefer using :class:`KDLoss` below which also allows weighting and optional
    feature alignment.
    """

    log_p = nn.functional.log_softmax(student_logits / temperature, dim=1)
    q = nn.functional.softmax(teacher_logits / temperature, dim=1)
    return nn.functional.kl_div(log_p, q, reduction="batchmean") * (temperature**2)


class KDLoss(nn.Module):
    """Composite knowledge distillation loss.

    Parameters
    ----------
    temperature:
        Softening factor applied to the logits.
    alpha:
        Weight of the distillation (teacher–student) term.
    beta:
        Optional weight for an auxiliary mean‑squared error between intermediate
        features of teacher and student.  Set to ``0`` to disable.
    """

    def __init__(self, temperature: float = 1.0, alpha: float = 1.0, beta: float = 0.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        *,
        student_feat: torch.Tensor | None = None,
        teacher_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.temperature
        log_p = nn.functional.log_softmax(student_logits / t, dim=1)
        q = nn.functional.softmax(teacher_logits / t, dim=1)
        loss = nn.functional.kl_div(log_p, q, reduction="batchmean") * (t * t)
        loss = loss * self.alpha
        if self.beta > 0 and student_feat is not None and teacher_feat is not None:
            loss = loss + self.beta * self.mse(student_feat, teacher_feat)
        return loss


__all__ = ["kd_loss", "KDLoss"]
