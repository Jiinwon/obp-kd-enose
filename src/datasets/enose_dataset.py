"""Dataset utilities for e-nose CSV logs."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


class ENoseDataset(Dataset):
    """Simple placeholder dataset for preprocessed e-nose signals.

    Parameters
    ----------
    data: Sequence
        Iterable of input windows (``C x T`` arrays).
    labels: Sequence
        Iterable of integer labels.
    """

    def __init__(self, data: Sequence, labels: Sequence) -> None:
        self.data = np.asarray(list(data))
        self.labels = np.asarray(list(labels))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.data[idx], int(self.labels[idx])
