#!/usr/bin/env python
"""
간단 미리보기:
python tools/preview_preprocess.py --registry configs/data_registry.yaml --dataset uci_drift --split train --pre common --cfg configs/preprocess/uci_drift.yaml
"""
import argparse

from src.datasets import UCIDriftDataset, TwinDataset, TempModDataset, HomeDataset, DynamicMixtureDataset, LongTermDataset, PulsesDataset, load_registry
from src.features.preprocessing import PreprocessPipeline

DS_MAP = {
    "uci_drift": UCIDriftDataset,
    "twin": TwinDataset,
    "tempmod": TempModDataset,
    "home": HomeDataset,
    "dynamic": DynamicMixtureDataset,
    "longterm": LongTermDataset,
    "pulses": PulsesDataset,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=list(DS_MAP.keys()))
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--pre", type=str, default="configs/preprocess/common.yaml")
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()

    reg = load_registry(args.registry)
    ds = DS_MAP[args.dataset](registry=reg, split=args.split)

    pipe = PreprocessPipeline.from_yaml(args.pre, args.cfg)
    items = [ds[i] for i in range(len(ds))]
    devs = [it[2].device_id for it in items]
    pipe.fit(items, train_device_ids=devs)

    X, y, meta = items[0]
    W = pipe.transform_and_window(X, device_id=meta.device_id)
    print(f"windows: {W.shape}  (N,C,W)")

if __name__ == "__main__":
    main()
