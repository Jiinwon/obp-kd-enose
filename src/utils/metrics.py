"""Metric utilities."""
from __future__ import annotations

import numpy as np


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute accuracy between two integer arrays."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float((pred == target).mean())
