"""Utilities for the UCI gas sensor array drift feature dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.datasets import load_svmlight_file


__all__ = [
    "BatchSplitConfig",
    "discover_batch_files",
    "load_feature_batches",
    "split_by_batches",
]


@dataclass(frozen=True)
class BatchSplitConfig:
    """Description of the batch-wise dataset partitioning."""

    train: Sequence[int]
    val: Sequence[int]
    test: Sequence[int]

    def all_batches(self) -> set[int]:
        """Return the set of all batch indices referenced in the config."""

        batches: set[int] = set()
        batches.update(int(b) for b in self.train)
        batches.update(int(b) for b in self.val)
        batches.update(int(b) for b in self.test)
        return batches


def discover_batch_files(data_root: Path) -> list[Path]:
    """Discover batch files sorted by their inferred batch number."""

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory '{data_root}' does not exist")
    if data_root.is_file():
        raise ValueError("'data_root' must be a directory containing batch files")

    batch_files = sorted(
        path
        for path in data_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".dat", ""}
    )
    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found in '{data_root}'. Expected LibSVM files named like 'batch1.dat'."
        )

    def key(path: Path) -> int:
        for token in path.stem.replace("-", "_").split("_"):
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                return int(digits)
        raise ValueError(f"Cannot infer batch index from filename '{path.name}'")

    return sorted(batch_files, key=key)


def load_feature_batches(data_root: Path, *, n_features: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the LibSVM-formatted batches into dense arrays."""

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    batch_ids: list[int] = []

    for batch_idx, batch_path in enumerate(discover_batch_files(data_root), start=1):
        data, labels = load_svmlight_file(batch_path, n_features=n_features)
        xs.append(data.toarray())
        ys.append(labels.astype(np.int64))
        batch_ids.extend([batch_idx] * data.shape[0])

    X = np.vstack(xs)
    y = np.concatenate(ys)
    batch_array = np.asarray(batch_ids, dtype=np.int64)
    return X, y, batch_array


def _mask_for_batches(batch_ids: np.ndarray, target_batches: Iterable[int]) -> np.ndarray:
    targets = {int(b) for b in target_batches}
    if not targets:
        raise ValueError("At least one batch index must be provided")
    return np.isin(batch_ids, list(targets))


def split_by_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_ids: np.ndarray,
    splits: BatchSplitConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return a dictionary with ``train``, ``val`` and ``test`` splits."""

    unique_batches = set(int(b) for b in np.unique(batch_ids))
    missing = splits.all_batches() - unique_batches
    if missing:
        raise ValueError(f"Requested batches {sorted(missing)} not present in dataset")

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, target_batches in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        mask = _mask_for_batches(batch_ids, target_batches)
        result[split_name] = (X[mask], y[mask])
    return result

