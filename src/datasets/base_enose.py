from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:
    from torch.utils.data import Dataset  # type: ignore
except Exception:  # torch 미설치 환경에서도 동작
    class Dataset:  # type: ignore
        pass


def load_registry(path: Union[str, Path]) -> Dict[str, Any]:
    """YAML 레지스트리를 로드한다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SampleMeta:
    dataset: str
    session_id: Union[str, int]
    device_id: Union[str, int, None]
    batch_id: Union[str, int, None]
    gas: Optional[str] = None
    conc: Optional[float] = None
    sampling_rate: Optional[float] = None
    file_path: Optional[str] = None


class BaseENoseDataset(Dataset):
    """
    공통 스키마 기반 e-nose 데이터셋.
    - 입력 파일은 'tidy' CSV 가정: time 열 + sensor_cols + (선택적) gas/conc/temp/humid/...
    - 1단계에서는 윈도잉/전처리 없이, 한 파일을 하나의 시퀀스 샘플로 반환한다.
    - 반환: (X, y, meta)
        X: np.ndarray, shape (C, T)
        y: dict {"gas": str|None, "conc": float|None}
        meta: SampleMeta
    """

    def __init__(
        self,
        *,
        registry: Dict[str, Any],
        dataset_key: str,
        split: str,
        root_override: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        if dataset_key not in registry:
            raise KeyError(f"dataset_key '{dataset_key}' not found in registry.")
        self.cfg = registry[dataset_key]
        self.dataset_key = dataset_key
        self.split = split

        self.root = Path(root_override) if root_override else Path(self.cfg["root"])
        self.mapping = self.cfg["mapping"]
        self.sensor_cols: List[str] = list(self.mapping.get("sensor_cols", []))
        if not self.sensor_cols:
            raise ValueError(f"{dataset_key}: mapping.sensor_cols가 비어있음")

        # 파일 인덱스 만들기
        files_cfg = self.cfg.get("files", {})
        pattern = files_cfg.get("pattern", "*.csv")
        all_files = sorted([str(p) for p in self.root.rglob(pattern)])

        # split 규칙 적용
        split_cfg = (self.cfg.get("splits") or {}).get(split, {})
        selected = self._apply_split_rules(all_files, split_cfg)

        # 인덱스 목록
        self.index: List[Tuple[str, SampleMeta]] = []
        for fp in selected:
            meta = self._build_meta_from_path(fp)
            self.index.append((fp, meta))

    def _apply_split_rules(self, all_files: List[str], split_cfg: Dict[str, Any]) -> List[str]:
        """registry의 split 규칙을 단순 적용한다."""
        # 가장 간단한 규칙: include_batches / devices / exclude_devices / ratio 등
        if not split_cfg:
            return all_files

        # batch/device는 파일명 또는 CSV 내부에서 해석해야 한다.
        # 1단계에서는 파일명에 'batch\d+' 또는 'B(\d+)' 패턴이 있다고 가정하고 필터링(없으면 전체).
        import re

        def batch_from_name(name: str) -> Optional[int]:
            m = re.search(r"batch(\d+)", name, re.IGNORECASE)
            if m:
                return int(m.group(1))
            m = re.search(r"day(\d+)", name, re.IGNORECASE)
            if m:
                return int(m.group(1))
            m = re.search(r"month(\d+)", name, re.IGNORECASE)
            if m:
                return int(m.group(1))
            return None

        def device_from_name(name: str) -> Optional[int]:
            m = re.search(r"[bB](\d+)", name)
            return int(m.group(1)) if m else None

        filtered = list(all_files)
        if "include_batches" in split_cfg:
            keep = set(split_cfg["include_batches"])
            filtered = [f for f in filtered if (batch := batch_from_name(f)) in keep]
        if "devices" in split_cfg:
            keep = set(split_cfg["devices"])
            filtered = [f for f in filtered if (dv := device_from_name(f)) in keep]
        if "exclude_devices" in split_cfg:
            ban = set(split_cfg["exclude_devices"])
            filtered = [f for f in filtered if (dv := device_from_name(f)) not in ban]

        # ratio 분할(시간순 정렬 가정)
        if "ratio" in split_cfg and split_cfg.get("time_order", False):
            r = split_cfg["ratio"]
            assert len(r) == 3, "ratio는 [train,val,test] 형태여야 함"
            n = len(filtered)
            t = int(n * r[0])
            v = t + int(n * r[1])
            if self.split == "train":
                filtered = filtered[:t]
            elif self.split == "val":
                filtered = filtered[t:v]
            elif self.split == "test":
                filtered = filtered[v:]

        # limit_first_n, range_index 등 간단 유틸
        if "limit_first_n" in split_cfg:
            filtered = filtered[: int(split_cfg["limit_first_n"])]
        if "range_index" in split_cfg:
            a, b = split_cfg["range_index"]
            filtered = filtered[a:b]

        return filtered

    def _build_meta_from_path(self, fp: str) -> SampleMeta:
        # 파일명에서 session_id 추출 (기본: 파일명)
        session_id = Path(fp).stem
        device_id = None
        batch_id = None
        sampling_rate = self.mapping.get("sampling_rate", None)
        return SampleMeta(
            dataset=self.dataset_key,
            session_id=session_id,
            device_id=device_id,
            batch_id=batch_id,
            sampling_rate=float(sampling_rate) if sampling_rate else None,
            file_path=fp,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fp, meta = self.index[i]
        # CSV 읽기(1단계). npz 등은 2단계에서 확장.
        df = pd.read_csv(fp)

        # 센서 행렬 (C,T)로 반환
        # time은 사용하지만 X에는 포함하지 않는다(센서만)
        sensor_cols = [c for c in self.sensor_cols if c in df.columns]
        if not sensor_cols:
            raise KeyError(f"{self.dataset_key}: 파일 {fp}에 sensor_cols가 없음")
        X = df[sensor_cols].to_numpy(dtype=np.float32).T  # (T,C) -> (C,T)

        # 라벨/부가 정보
        gas_col = self.mapping.get("gas", None)
        conc_col = self.mapping.get("conc", None)
        gas = str(df[gas_col].iloc[0]) if (gas_col and gas_col in df.columns) else None
        conc = float(df[conc_col].iloc[0]) if (conc_col and conc_col in df.columns) else None

        y = {"gas": gas, "conc": conc}
        return X, y, meta
