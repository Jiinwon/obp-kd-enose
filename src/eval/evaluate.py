"""Model evaluation utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.metrics import accuracy


def evaluate(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute basic metrics for predictions."""
    return {"acc": accuracy(pred, target)}
