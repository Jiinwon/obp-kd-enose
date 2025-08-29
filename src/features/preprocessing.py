"""Signal preprocessing utilities."""
from __future__ import annotations

from typing import Iterable, List


def _to_list(x: Iterable) -> List[float]:
    return list(x)


def warmup_cut(x: Iterable, n: int = 5):
    """Remove an initial warm-up period from the signal."""
    return _to_list(x)[n:]


def center(x: Iterable):
    """Mean-center the signal."""
    data = _to_list(x)
    mean = sum(data) / len(data) if data else 0.0
    return [v - mean for v in data]
