from __future__ import annotations

import numpy as np

from src.datasets import build_dataset
from src.features.preprocessing import PreprocessPipeline


def _aux(meta):
    return {"temp": meta.temp, "humid": meta.humid}


def test_standardization_freeze(tmp_path):
    dataset = build_dataset(
        "uci_drift",
        "train",
        "configs/data_registry.yaml",
        synthetic_ok=True,
    )
    pipeline = PreprocessPipeline.from_yaml(
        "configs/preprocess/common.yaml", "configs/preprocess/uci_drift.yaml"
    )
    train_samples = [dataset[i] for i in range(len(dataset))]
    device_ids = [meta.device_id for (_, _, meta) in train_samples]
    pipeline.fit(train_samples, train_device_ids=device_ids, aux_provider=_aux)
    mean_before = pipeline.std_state.mean.copy()
    std_before = pipeline.std_state.std.copy()
    for X, _, meta in train_samples:
        pipeline.transform_and_window(X, device_id=meta.device_id, aux=_aux(meta))
    np.testing.assert_allclose(pipeline.std_state.mean, mean_before)
    np.testing.assert_allclose(pipeline.std_state.std, std_before)
