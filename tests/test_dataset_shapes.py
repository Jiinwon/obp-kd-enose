from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

from src.datasets.base_enose import BaseENoseDataset, load_registry


def _make_csv(dirpath: Path, name: str, nT: int = 50, C: int = 4):
    t = np.arange(nT)
    data = {"time": t, "gas": ["CO"] * nT, "ppm": [100.0] * nT}
    for c in range(C):
        data[f"s{c}"] = np.random.rand(nT)
    df = pd.DataFrame(data)
    fp = dirpath / name
    df.to_csv(fp, index=False)
    return fp


def test_dataset_returns_CT():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "toy"
        root.mkdir(parents=True, exist_ok=True)
        _make_csv(root, "sample.csv", nT=50, C=4)

        reg_text = f"""toy:
  root: "{root.as_posix()}"
  files: {{pattern: "*.csv"}}
  mapping:
    time: "time"
    sensor_cols: ["s0","s1","s2","s3"]
    gas: "gas"
    conc: "ppm"
    temp: null
    humid: null
    stage: null
    batch_id: null
    device_id: 0
    session_id: "file_id"
    sampling_rate: 100.0
  splits:
    train: {{}}
"""
        reg_p = Path(td) / "reg.yaml"
        reg_p.write_text(reg_text, encoding="utf-8")
        reg = load_registry(reg_p)

        ds = BaseENoseDataset(registry=reg, dataset_key="toy", split="train")
        assert len(ds) == 1
        X, y, meta = ds[0]
        assert X.shape[0] == 4  # C
        assert X.shape[1] == 50  # T
        assert y["gas"] == "CO"
        assert y["conc"] == 100.0
        assert meta.dataset == "toy"
