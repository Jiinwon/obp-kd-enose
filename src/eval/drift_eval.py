"""Evaluation utilities for sensor drift scenarios."""
from __future__ import annotations

from typing import Sequence, Tuple

from .evaluate import evaluate


def split_by_time(series: Sequence, ratio: float = 0.5) -> Tuple[Sequence, Sequence]:
    """Split ``series`` into early/late segments according to ``ratio``."""

    mid = int(len(series) * ratio)
    return series[:mid], series[mid:]


def split_by_rht(
    values: Sequence,
    rh: Sequence[float],
    temp: Sequence[float],
    rh_threshold: float,
    t_threshold: float,
) -> Tuple[Sequence, Sequence]:
    """Split samples depending on RH/T thresholds."""

    early, late = [], []
    for v, r, t in zip(values, rh, temp):
        if r <= rh_threshold and t <= t_threshold:
            early.append(v)
        else:
            late.append(v)
    return early, late


def delta_auc(pred_a, target_a, pred_b, target_b, *, prob_a=None, prob_b=None):
    """Compute the change in ROC‑AUC between two splits."""

    metrics_a = evaluate(pred_a, target_a, prob=prob_a)
    metrics_b = evaluate(pred_b, target_b, prob=prob_b)
    auc_a = metrics_a.get("roc_auc", 0.0)
    auc_b = metrics_b.get("roc_auc", 0.0)
    return auc_b - auc_a


__all__ = ["split_by_time", "split_by_rht", "delta_auc"]
