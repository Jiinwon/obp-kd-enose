"""Loader for the UCI Gas Sensor Array Drift dataset.

Only a very small portion of the original functionality is required for the
unit tests, however the implementation below demonstrates how the dataset can
be prepared for experiments.  The loader assumes that the data has been
pre‑processed into ``.npz`` files containing at least ``x`` and ``y`` arrays and
optionally ``batch`` indices and ``time`` stamps.  To keep dependencies small
only the standard library and (optionally) :mod:`torch` are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Sequence, List
import json

try:
    import numpy as _np  # Used only if available
except Exception:  # pragma: no cover - numpy may not be installed
    _np = None  # type: ignore

try:  # torch is optional
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore


class UCIDriftDataset(Dataset):
    """Dataset providing time/batch wise splits.

    The constructor expects a path to an ``.npz`` file.  The arrays ``x`` and
    ``y`` are mandatory.  When ``split_type`` is ``"time"`` a ``time`` array is
    required, for ``split_type="batch"`` a ``batch`` array must be present.  The
    ``split`` argument controls whether the first or second half of the ordered
    data is returned.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        split_type: str = "time",
    ) -> None:
        if _np is None:
            raise ImportError("numpy is required to load the UCI dataset")

        arr = _np.load(path)
        x = arr["x"]
        y = arr["y"]

        if split_type == "time":
            order = _np.argsort(arr["time"])
        elif split_type == "batch":
            order = _np.argsort(arr["batch"])
        else:
            raise ValueError("split_type must be 'time' or 'batch'")

        x = x[order]
        y = y[order]

        mid = len(x) // 2
        if split == "train":
            self.x = x[:mid]
            self.y = y[:mid]
        else:
            self.x = x[mid:]
            self.y = y[mid:]

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple:
        sample_x = self.x[idx]
        sample_y = int(self.y[idx])
        if torch is not None:
            sample_x = torch.tensor(sample_x, dtype=torch.float32)
        return sample_x, sample_y


__all__ = ["UCIDriftDataset"]

