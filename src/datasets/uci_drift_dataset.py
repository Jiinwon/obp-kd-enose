"""Loader for the UCI gas sensor drift dataset."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


class UCIDriftDataset(Dataset):
    """Very small placeholder dataset for testing purposes."""

    def __init__(self, data: Sequence, labels: Sequence) -> None:
        self.data = np.asarray(list(data))
        self.labels = np.asarray(list(labels))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.data[idx], int(self.labels[idx])
