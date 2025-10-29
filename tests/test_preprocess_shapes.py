import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from src.datasets.base_enose import BaseENoseDataset, load_registry
from src.features.preprocessing import PreprocessPipeline

def _make_csv(dirpath: Path, name: str, nT: int = 257, C: int = 8):
    t = np.arange(nT)
    data = { "time": t, "gas": ["CO"]*nT, "ppm": [100.0]*nT }
    for c in range(C):
        data[f"s{c}"] = 2.0 + 0.1*np.sin(0.01*t + 0.1*c) + 0.01*np.random.randn(nT)
    df = pd.DataFrame(data)
    fp = dirpath / name
    df.to_csv(fp, index=False)
    return fp

def test_windowing_counts_and_shapes():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "toy"
        root.mkdir(parents=True, exist_ok=True)
        _make_csv(root, "sample.csv")

        reg_text = f"""\
toy:
  root: "{root.as_posix()}"
  files: {{pattern: "*.csv"}}
  mapping:
    time: "time"
    sensor_cols: ["s0","s1","s2","s3","s4","s5","s6","s7"]
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

        # 파이프라인(공통 설정 inline)
        class _Cfg:
            window={"size":64,"stride":16,"drop_last":True}
            rr0={"enabled":True,"mode":"ratio","baseline":{"method":"first_k","k":8}}
            detrend={"enabled":True,"method":"ema","ema_alpha":0.1}
            standardize={"enabled":True,"method":"zscore","eps":1e-6,"per_device":False}
            temp_humid={"enabled":False}
        pipe = PreprocessPipeline(_Cfg)

        # 학습
        train_iter = [ds[i] for i in range(len(ds))]
        devs = [s[2].device_id for s in train_iter]
        pipe.fit(train_iter, train_device_ids=devs)

        # 변환+윈도
        X, y, meta = ds[0]
        W = pipe.transform_and_window(X, device_id=meta.device_id)
        assert W.ndim == 3 and W.shape[1] == 8 and W.shape[2] == 64
        # 창 개수 확인: N = floor((T-W)/stride) = floor((257-64)/16) = 12 (stride가 딱 맞지 않아 마지막 윈도는 drop)
        assert W.shape[0] == 12
