"""Compatibility layer for prior table utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .prior_pipeline import PriorTable, build_label_to_prior, load_prior_table


def load_priors(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load legacy JSON prior files mapping class names to vectors."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = list(data.keys())
    matrix = np.vstack([np.asarray(data[c], dtype=np.float32) for c in classes]) if classes else np.zeros((0, 0), dtype=np.float32)
    return matrix, classes


def load_prior(csv_path: str | Path) -> PriorTable:
    """Backward-compatible loader that wraps :func:`load_prior_table`."""

    return load_prior_table(csv_path)


def labels_to_prior(
    labels: Sequence[str],
    prior_csv: str | Path,
    voc_map: str | Path,
    *,
    default: str = "mean",
) -> np.ndarray:
    """Load the prior table and map labels in a single call."""

    table = load_prior_table(prior_csv)
    return build_label_to_prior(labels, table, voc_map, default=default)


__all__ = [
    "PriorTable",
    "load_prior",
    "labels_to_prior",
    "build_label_to_prior",
    "load_prior_table",
    "load_priors",
]
