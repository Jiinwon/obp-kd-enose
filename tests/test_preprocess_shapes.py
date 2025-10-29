from __future__ import annotations

import math

from src.datasets import build_dataset
from src.features.preprocessing import PreprocessPipeline


def _aux(meta):
    return {"temp": meta.temp, "humid": meta.humid}


def test_window_shape_formula():
    dataset = build_dataset(
        "uci_drift",
        "train",
        "configs/data_registry.yaml",
        synthetic_ok=True,
    )
    pipeline = PreprocessPipeline.from_yaml(
        "configs/preprocess/common.yaml", "configs/preprocess/uci_drift.yaml"
    )
    samples = [dataset[i] for i in range(len(dataset))]
    device_ids = [meta.device_id for (_, _, meta) in samples]
    pipeline.fit(samples, train_device_ids=device_ids, aux_provider=_aux)
    X, _, meta = samples[0]
    windows = pipeline.transform_and_window(X, device_id=meta.device_id, aux=_aux(meta))
    size = pipeline.cfg.window["size"]
    stride = pipeline.cfg.window["stride"]
    expected = math.floor((X.shape[1] - size) / stride) + 1 if X.shape[1] >= size else 0
    assert windows.shape[0] == expected
    if expected > 0:
        assert windows.shape[1] == X.shape[0]
        assert windows.shape[2] == size
