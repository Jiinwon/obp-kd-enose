"""Signal preprocessing utilities."""
from __future__ import annotations

import numpy as np


def warmup_cut(x: np.ndarray, n: int = 5) -> np.ndarray:
    """Remove an initial warm-up period from the signal."""
    return np.asarray(x)[n:]


def center(x: np.ndarray) -> np.ndarray:
    """Mean-center the signal."""
    x = np.asarray(x)
    return x - x.mean()
