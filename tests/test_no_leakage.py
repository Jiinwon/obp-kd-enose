import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from src.datasets.base_enose import BaseENoseDataset, load_registry
from src.features.preprocessing import PreprocessPipeline

def _make_csv(dirpath: Path, name: str, start: int, nT: int = 200, C: int = 4):
    t = np.arange(start, start + nT)
    data = { "time": t, "gas": ["CO"]*nT, "ppm": [100.0]*nT }
    for c in range(C):
        # 서로 다른 분포를 주어 train/test가 구분되게 함
        base = 1.0 if "train" in name else 3.0
        data[f"s{c}"] = base + 0.01 * np.random.randn(nT)
    df = pd.DataFrame(data)
    fp = dirpath / name
    df.to_csv(fp, index=False)
    return fp

def test_no_leakage_and_stats_freeze():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "toy"
        root.mkdir(parents=True, exist_ok=True)
        _make_csv(root, "batch1_train.csv", start=0)
        _make_csv(root, "batch2_val.csv", start=1000)

        reg_text = f"""\
toy:
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
    val:   {{}}
"""
        reg_p = Path(td) / "reg.yaml"
        reg_p.write_text(reg_text, encoding="utf-8")
        reg = load_registry(reg_p)

        ds_train = BaseENoseDataset(registry=reg, dataset_key="toy", split="train")
        ds_val   = BaseENoseDataset(registry=reg, dataset_key="toy", split="val")

        # 파이프라인 로드(공통+데이터셋 설정 대체)
        common = {
          "window":{"size":64,"stride":32,"drop_last":True},
          "rr0":{"enabled":True,"mode":"ratio","baseline":{"method":"first_k","k":10}},
          "detrend":{"enabled":True,"method":"ema","ema_alpha":0.05},
          "standardize":{"enabled":True,"method":"zscore","eps":1e-6,"per_device":False},
          "temp_humid":{"enabled":False}
        }
        cfg_path = Path(td) / "tmp_common.yaml"
        cfg_path.write_text(yaml_dump:=__import__("yaml").dump(common), encoding="utf-8")

        pipe = PreprocessPipeline.from_yaml(cfg_path, cfg_path)

        # fit(train)
        train_iter = [ds_train[i] for i in range(len(ds_train))]
        train_devs = [s[2].device_id for s in train_iter]
        pipe.fit(train_iter, train_device_ids=train_devs, aux_provider=None)

        # 학습 통계 저장
        mean_before = pipe.std_state.mean.copy()
        std_before  = pipe.std_state.std.copy()

        # transform(val)
        val_iter = [ds_val[i] for i in range(len(ds_val))]
        for X, y, meta in val_iter:
            W = pipe.transform_and_window(X, device_id=meta.device_id, aux=None)
            assert W.ndim == 3 and W.shape[1] == 4 and W.shape[2] == 64

        # 통계가 변하지 않는지 확인(프리징)
        np.testing.assert_allclose(mean_before, pipe.std_state.mean)
        np.testing.assert_allclose(std_before,  pipe.std_state.std)
