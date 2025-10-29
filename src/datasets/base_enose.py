"""Dataset utilities for e-nose time series."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:  # pragma: no cover - optional dependency
    from torch.utils.data import Dataset  # type: ignore
except Exception:  # pragma: no cover
    class Dataset:  # type: ignore
        """Fallback Dataset stub when torch is unavailable."""

        pass


def load_data_registry(path: str | Path) -> Dict[str, Any]:
    """Load the YAML registry describing all datasets."""

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_registry(path: str | Path) -> Dict[str, Any]:
    """Backward-compatible alias for :func:`load_data_registry`."""

    return load_data_registry(path)


@dataclass
class SampleMeta:
    """Metadata for a single e-nose recording adhering to the common schema."""

    dataset: str
    session_id: str
    device_id: Optional[str]
    sensor_id: Optional[str]
    stage: Optional[str]
    gas: Optional[str]
    conc: Optional[float]
    batch_id: Optional[str]
    sampling_rate: Optional[float]
    file_path: str
    time: np.ndarray
    temp: Optional[np.ndarray]
    humid: Optional[np.ndarray]

    def with_aux(self, *, temp: Optional[np.ndarray], humid: Optional[np.ndarray]) -> "SampleMeta":
        return replace(self, temp=temp, humid=humid)


class BaseENoseDataset(Dataset):
    """Dataset returning raw sessions (no windowing)."""

    def __init__(
        self,
        *,
        registry: Dict[str, Any],
        dataset_key: str,
        split: str,
        root_override: Optional[Union[str, Path]] = None,
        synthetic_ok: bool = False,
    ) -> None:
        super().__init__()
        if dataset_key not in registry:
            raise KeyError(f"dataset_key '{dataset_key}' not found in registry")
        self.registry_entry = registry[dataset_key]
        self.dataset_key = dataset_key
        self.split = split
        self.mapping = self.registry_entry.get("mapping", {})
        self.sensor_cols: List[str] = list(self.mapping.get("sensor_cols", []))
        if not self.sensor_cols:
            raise ValueError(f"{dataset_key}: mapping.sensor_cols must be defined")
        self.root = Path(root_override) if root_override else Path(self.registry_entry["root"])
        self.file_pattern = self.registry_entry.get("files", {}).get("pattern", "**/*.csv")
        if synthetic_ok:
            ensure_synthetic_dataset(self, self.registry_entry)
        all_files = sorted(str(p) for p in self.root.glob(self.file_pattern) if p.is_file())
        split_cfg = (self.registry_entry.get("splits") or {}).get(split, {})
        selected = self._apply_split_rules(all_files, split_cfg)
        self.index: List[Tuple[str, SampleMeta]] = []
        for path in selected:
            meta = self._build_meta_from_path(path)
            self.index.append((path, meta))

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_numeric(token: str) -> Optional[int]:
        import re

        match = re.search(r"(\d+)", token)
        return int(match.group(1)) if match else None

    def _apply_split_rules(self, files: Sequence[str], split_cfg: Dict[str, Any]) -> List[str]:
        if not split_cfg:
            return list(files)
        selected = list(files)
        if "forward_batches" in split_cfg:
            keep = {int(x) for x in split_cfg["forward_batches"]}
            filtered: List[str] = []
            for path in selected:
                num = self._parse_numeric(Path(path).stem)
                if num is not None and num in keep:
                    filtered.append(path)
            selected = filtered
        if "devices" in split_cfg:
            keep = {int(x) for x in split_cfg["devices"]}
            filtered = []
            for path in selected:
                num = self._parse_numeric(Path(path).stem.split("device")[-1])
                if num is not None and num in keep:
                    filtered.append(path)
            selected = filtered
        if "include_cycles" in split_cfg:
            include = {int(x) for x in split_cfg["include_cycles"]}
            selected = [p for p in selected if self._parse_numeric(Path(p).stem.split("cycle")[-1]) in include]
        if "include_days" in split_cfg:
            include = {int(x) for x in split_cfg["include_days"]}
            selected = [p for p in selected if self._parse_numeric(Path(p).stem.split("day")[-1]) in include]
        if "include_mixtures" in split_cfg:
            include = {str(x) for x in split_cfg["include_mixtures"]}
            selected = [p for p in selected if any(tag in Path(p).stem for tag in include)]
        if "include_batches" in split_cfg:
            include = {int(x) for x in split_cfg["include_batches"]}
            selected = [p for p in selected if self._parse_numeric(Path(p).stem.split("batch")[-1]) in include]
        if "include_pulses" in split_cfg:
            include = {str(x) for x in split_cfg["include_pulses"]}
            selected = [p for p in selected if any(tag in Path(p).stem for tag in include)]
        if "limit_first_n" in split_cfg:
            selected = selected[: int(split_cfg["limit_first_n"])]
        return selected

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _build_meta_from_path(self, fp: str) -> SampleMeta:
        session_id = Path(fp).stem
        sampling_rate = self.mapping.get("sampling_rate")
        return SampleMeta(
            dataset=self.dataset_key,
            session_id=session_id,
            device_id=None,
            sensor_id=None,
            stage=None,
            gas=None,
            conc=None,
            batch_id=None,
            sampling_rate=float(sampling_rate) if sampling_rate else None,
            file_path=fp,
            time=np.empty(0, dtype=np.float32),
            temp=None,
            humid=None,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any], SampleMeta]:
        fp, meta = self.index[idx]
        df = pd.read_csv(fp)
        sensor_cols = [c for c in self.sensor_cols if c in df.columns]
        if not sensor_cols:
            raise KeyError(f"{self.dataset_key}: sensors {self.sensor_cols} not found in {fp}")
        X = df[sensor_cols].to_numpy(dtype=np.float32).T
        mapping = self.mapping
        time_col = mapping.get("time")
        time = df[time_col].to_numpy(dtype=np.float32) if time_col and time_col in df.columns else np.arange(X.shape[1], dtype=np.float32)
        gas_col = mapping.get("gas")
        gas = str(df[gas_col].iloc[0]) if gas_col and gas_col in df.columns else None
        conc_col = mapping.get("conc")
        conc = float(df[conc_col].iloc[0]) if conc_col and conc_col in df.columns else None
        temp_col = mapping.get("temp")
        temp = df[temp_col].to_numpy(dtype=np.float32) if temp_col and temp_col in df.columns else None
        humid_col = mapping.get("humid")
        humid = df[humid_col].to_numpy(dtype=np.float32) if humid_col and humid_col in df.columns else None
        device_col = mapping.get("device_id")
        device_id = str(df[device_col].iloc[0]) if device_col and device_col in df.columns else meta.device_id
        stage_col = mapping.get("stage")
        stage = str(df[stage_col].iloc[0]) if stage_col and stage_col in df.columns else None
        batch_col = mapping.get("batch_id")
        batch_id = str(df[batch_col].iloc[0]) if batch_col and batch_col in df.columns else None
        meta = meta.with_aux(temp=temp, humid=humid)
        meta = replace(meta, device_id=device_id, stage=stage, gas=gas, conc=conc, batch_id=batch_id, time=time)
        y = {"gas": gas, "conc": conc}
        return X, y, meta


def build_dataset(
    dataset_key: str,
    split: str,
    registry_path: str,
    *,
    root_override: Optional[str] = None,
    synthetic_ok: bool = False,
) -> BaseENoseDataset:
    """Instantiate a dataset from the registry."""

    registry = load_data_registry(registry_path)
    dataset = BaseENoseDataset(
        registry=registry,
        dataset_key=dataset_key,
        split=split,
        root_override=root_override,
        synthetic_ok=synthetic_ok,
    )
    return dataset


def ensure_synthetic_dataset(dataset: BaseENoseDataset, registry_entry: Dict[str, Any]) -> None:
    """Create synthetic CSV files when raw data is unavailable."""

    root = dataset.root
    pattern = registry_entry.get("files", {}).get("pattern", "**/*.csv")
    existing = list(root.glob(pattern))
    synth_cfg = registry_entry.get("synthetic", {})
    if existing or not synth_cfg.get("enabled", False):
        return
    root.mkdir(parents=True, exist_ok=True)
    per_split: Dict[str, int] = synth_cfg.get("per_split", {})
    sensors = synth_cfg.get("sensors", len(dataset.sensor_cols))
    length = int(synth_cfg.get("length", 200))
    gases = synth_cfg.get("gases", ["UNKNOWN"])
    rng = np.random.default_rng(0)
    for split, count in per_split.items():
        split_cfg = (registry_entry.get("splits") or {}).get(split, {})
        batch_values = split_cfg.get("forward_batches") or split_cfg.get("include_batches")
        if batch_values:
            items = list(batch_values)
        else:
            items = list(range(1, int(count) + 1))
        if len(items) < int(count):
            items.extend(range(len(items) + 1, int(count) + 1))
        for idx, batch_value in enumerate(items[: int(count)]):
            time = np.arange(length, dtype=np.float32)
            signal = rng.normal(size=(sensors, length)).cumsum(axis=1).astype(np.float32)
            gas = gases[idx % len(gases)]
            conc = float(idx + 1)
            temp = 25 + 2 * np.sin(time / 50)
            humid = 40 + 5 * np.cos(time / 60)
            data: Dict[str, Any] = {}
            mapping = dataset.mapping
            data[mapping.get("time", "time")] = time
            for i, col in enumerate(dataset.sensor_cols[:sensors]):
                data[col] = signal[i]
            if mapping.get("gas"):
                data[mapping["gas"]] = gas
            if mapping.get("conc"):
                data[mapping["conc"]] = conc
            if mapping.get("temp"):
                data[mapping["temp"]] = temp
            if mapping.get("humid"):
                data[mapping["humid"]] = humid
            if mapping.get("device_id"):
                data[mapping["device_id"]] = f"dev{idx+1}"
            if mapping.get("session_id"):
                data[mapping["session_id"]] = f"{split}_session{idx+1}"
            if mapping.get("stage"):
                data[mapping["stage"]] = f"{split}_stage"
            if mapping.get("batch_id"):
                data[mapping["batch_id"]] = f"batch{batch_value}"
            df = pd.DataFrame(data)
            out_dir = root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{dataset.dataset_key}_{split}_batch{batch_value}_device{idx+1}.csv"
            df.to_csv(out_dir / filename, index=False)


__all__ = [
    "SampleMeta",
    "BaseENoseDataset",
    "build_dataset",
    "ensure_synthetic_dataset",
    "load_data_registry",
    "load_registry",
]
