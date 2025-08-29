"""Dataset utilities for processed e‑nose signals.

The original project combines a number of preprocessing steps and windowing
operations before the data is fed into a model.  For the unit tests we only
require a very small subset of the functionality, but the implementation below
is intentionally feature rich so that it mirrors the behaviour of the real
code base.

The :class:`ENoseDataset` expects raw signals together with class labels,
optional TVOC (parts‑per‑million) targets and a prior matrix.  During
initialisation each signal is preprocessed using
``src.features.preprocessing.preprocess_signal`` which produces a number of
sliding windows.  Each window becomes one sample returned by the dataset.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List

try:  # torch is an optional dependency in the tests
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch might be missing
    torch = None  # type: ignore
    Dataset = object  # type: ignore

from ..features.preprocessing import preprocess_signal


class ENoseDataset(Dataset):
    """Dataset exposing ``(x, y_cls, y_ppm, prior_vec)`` tuples.

    Parameters
    ----------
    signals:
        Sequence of raw time‑series.
    y_cls:
        Sequence of integer class labels.
    y_ppm:
        Optional sequence of TVOC concentration targets.  If omitted zeros are
        used.
    prior_matrix:
        Optional matrix of shape ``(num_classes, prior_dim)`` providing a prior
        vector for each class.  The entry for the ground truth class is returned
        with every sample.
    preprocess_cfg:
        Configuration dictionary forwarded to
        :func:`preprocess_signal`.
    """

    def __init__(
        self,
        signals: Sequence[Iterable[float]],
        y_cls: Sequence[int],
        y_ppm: Sequence[float] | None = None,
        *,
        prior_matrix: Sequence[Sequence[float]] | None = None,
        preprocess_cfg: dict | None = None,
    ) -> None:
        if y_ppm is None:
            y_ppm = [0.0] * len(y_cls)

        self.x: List[List[List[float]]] = []
        self.y_cls: List[int] = []
        self.y_ppm: List[float] = []
        self.priors: List[List[float]] = []

        for sig, cls, ppm in zip(signals, y_cls, y_ppm):
            windows = preprocess_signal(sig, cfg=preprocess_cfg or {})
            for win in windows:
                self.x.append(win)
                self.y_cls.append(int(cls))
                self.y_ppm.append(float(ppm))
                if prior_matrix is not None:
                    self.priors.append(list(prior_matrix[int(cls)]))

        self.has_prior = bool(self.priors)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple:
        x = self.x[idx]
        y_cls = self.y_cls[idx]
        y_ppm = self.y_ppm[idx]
        prior = self.priors[idx] if self.has_prior else []
        if torch is not None:
            x = torch.tensor(x, dtype=torch.float32)
            prior = torch.tensor(prior, dtype=torch.float32)
            y_cls = int(y_cls)
            y_ppm = float(y_ppm)
        return x, y_cls, y_ppm, prior


__all__ = ["ENoseDataset"]

