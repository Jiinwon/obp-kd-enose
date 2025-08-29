"""Model evaluation utilities."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple
import math

try:  # optional heavy dependency
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
except Exception:  # pragma: no cover - sklearn might be missing
    roc_auc_score = f1_score = confusion_matrix = None  # type: ignore

from ..utils.metrics import accuracy


def _confusion_matrix(pred: Sequence[int], target: Sequence[int], num_classes: int):
    mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for p, t in zip(pred, target):
        mat[t][p] += 1
    return mat


def _f1_scores(pred: Sequence[int], target: Sequence[int], num_classes: int) -> Tuple[Sequence[float], float]:
    if f1_score is not None:
        # micro average is computed by treating the multi-class confusion matrix as binary
        f1_micro = f1_score(target, pred, average="micro")
        f1_per_class = f1_score(target, pred, average=None)
        return f1_per_class, f1_micro

    # Fallback implementation without sklearn
    cm = _confusion_matrix(pred, target, num_classes)
    f1_per_class = []
    total_tp = total_fp = total_fn = 0
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][r] for r in range(num_classes) if r != c)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_per_class.append(f1)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0.0
    return f1_per_class, f1_micro


def _roc_auc(prob: Sequence[Sequence[float]], target: Sequence[int]) -> float | None:
    if roc_auc_score is None:
        return None
    try:
        return float(roc_auc_score(target, prob, multi_class="ovr"))
    except Exception:  # pragma: no cover - sklearn may raise
        return None


def rmse(pred: Sequence[float], target: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred))


def evaluate(
    pred: Sequence[int],
    target: Sequence[int],
    *,
    prob: Sequence[Sequence[float]] | None = None,
    tvoc_pred: Sequence[float] | None = None,
    tvoc_target: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Compute a set of metrics for classification/regression outputs."""

    pred = list(pred)
    target = list(target)
    num_classes = max(max(pred, default=0), max(target, default=0)) + 1

    metrics: Dict[str, float] = {"acc": accuracy(pred, target)}

    f1_c, f1_micro = _f1_scores(pred, target, num_classes)
    metrics.update({f"f1_{i}": f for i, f in enumerate(f1_c)})
    metrics["f1_micro"] = f1_micro

    cm = _confusion_matrix(pred, target, num_classes)
    # flatten confusion matrix for ease of logging
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            metrics[f"cm_{i}_{j}"] = float(val)

    if prob is not None:
        auc = _roc_auc(prob, target)
        if auc is not None:
            metrics["roc_auc"] = auc

    if tvoc_pred is not None and tvoc_target is not None:
        metrics["tvoc_rmse"] = rmse(tvoc_pred, tvoc_target)

    return metrics


__all__ = ["evaluate", "rmse"]
