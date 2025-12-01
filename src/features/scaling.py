"""Feature scaling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


__all__ = ["ScaledDatasets", "fit_minmax", "save_scaler", "load_scaler"]


@dataclass
class ScaledDatasets:
    """Container holding scaled splits."""

    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    scaler: MinMaxScaler


def fit_minmax(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> ScaledDatasets:
    """Fit a MinMax scaler on the train split and transform all splits."""

    if "train" not in datasets:
        raise KeyError("'train' split missing from datasets")
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train, y_train = datasets["train"]
    scaler.fit(X_train)

    def _transform(split: str) -> tuple[np.ndarray, np.ndarray]:
        X, y = datasets[split]
        return scaler.transform(X), y

    scaled = {
        split: _transform(split)
        for split in ("train", "val", "test")
        if split in datasets
    }
    return ScaledDatasets(
        train=scaled.get("train", datasets["train"]),
        val=scaled.get("val", datasets.get("val", (np.empty((0, X_train.shape[1])), np.empty(0)))),
        test=scaled.get("test", datasets.get("test", (np.empty((0, X_train.shape[1])), np.empty(0)))),
        scaler=scaler,
    )


def save_scaler(scaler: MinMaxScaler, path: Path) -> None:
    """Persist the fitted scaler using joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> MinMaxScaler:
    """Load a MinMax scaler from disk."""

    return joblib.load(path)

