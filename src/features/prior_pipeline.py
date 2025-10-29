"""Utilities for docking prior tables."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml


@dataclass
class PriorTable:
    """Structured representation of the docking prior table."""

    table: pd.DataFrame
    obps: List[str]
    metrics: List[str]

    @property
    def prior_dim(self) -> int:
        return len(self.obps) * len(self.metrics)

    def lookup(self, label: str, *, default: np.ndarray) -> np.ndarray:
        if label in self.table.index:
            return self.table.loc[label].to_numpy(dtype=np.float32)
        return default.astype(np.float32)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_obp_list(path: Path) -> List[str]:
    data = _load_yaml(path)
    return list(data.get("obps", []))


def _load_voc_map(path: Path) -> Dict[str, str]:
    data = _load_yaml(path)
    mapping = data.get("map", {})
    return {str(k): str(v) for k, v in mapping.items()}


def _read_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported docking log format: {path}")


def _load_logs(glob_pattern: str) -> List[pd.DataFrame]:
    paths = sorted(Path().glob(glob_pattern))
    dfs: List[pd.DataFrame] = []
    for path in paths:
        try:
            dfs.append(_read_log(path))
        except Exception as exc:
            print(f"[prior] Failed to read {path}: {exc}")
    return dfs


def _flip_affinity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "affinity" in df.columns:
        df["affinity"] = -df["affinity"].astype(float)
    return df


def _compute_metrics(group: pd.DataFrame) -> Dict[str, float]:
    affinity = group["affinity"].astype(float)
    metrics = {
        "min_affinity": float(affinity.max()),
        "p25_affinity": float(np.percentile(affinity, 25)) if len(affinity) else math.nan,
        "median_affinity": float(np.median(affinity)) if len(affinity) else math.nan,
        "p75_affinity": float(np.percentile(affinity, 75)) if len(affinity) else math.nan,
        "mean_affinity": float(affinity.mean()) if len(affinity) else math.nan,
        "std_affinity": float(affinity.std(ddof=0)) if len(affinity) else math.nan,
    }
    if "rmsd" in group.columns:
        metrics["best_rmsd"] = float(group["rmsd"].astype(float).min())
    else:
        metrics["best_rmsd"] = math.nan
    top3 = affinity.nlargest(min(len(affinity), 3))
    metrics["top3_mean_affinity"] = float(top3.mean()) if len(top3) else math.nan
    return metrics


def _fill_and_stack(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    obps: Sequence[str],
    metrics: Sequence[str],
    *,
    fill: float,
) -> pd.DataFrame:
    rows: Dict[str, Dict[str, float]] = {}
    for ligand, obp_stats in agg.items():
        row: Dict[str, float] = {}
        for obp in obps:
            stats = obp_stats.get(obp, {})
            for metric in metrics:
                row[f"{obp}__{metric}"] = float(stats.get(metric, fill))
        rows[ligand] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.empty:
        columns = [f"{obp}__{metric}" for obp in obps for metric in metrics]
        df = pd.DataFrame(columns=columns, dtype=float)
    df.sort_index(inplace=True)
    return df


def _zscore(df: pd.DataFrame, eps: float, fill: str) -> pd.DataFrame:
    if df.empty:
        return df
    filled = df.fillna(df.mean() if fill == "mean" else 0.0)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0, ddof=0).replace(0.0, eps)
    return (filled - mean) / std


def _aggregate_logs(dfs: Iterable[pd.DataFrame], obps: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for df in dfs:
        if df.empty:
            continue
        if not {"ligand", "obp", "affinity"}.issubset(df.columns):
            continue
        df = _flip_affinity(df)
        for (ligand, obp), group in df.groupby(["ligand", "obp"]):
            metrics = _compute_metrics(group)
            results.setdefault(ligand, {})[str(obp)] = metrics
    return results


def _build_empty_prior(obps: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    columns = [f"{obp}__{metric}" for obp in obps for metric in metrics]
    return pd.DataFrame(columns=columns, dtype=float)


def _write_qc(path: Path, table: pd.DataFrame, obps: Sequence[str], metrics: Sequence[str]) -> None:
    qc = {
        "obps": list(obps),
        "metrics": list(metrics),
        "rows": int(table.shape[0]),
        "columns": int(table.shape[1]),
        "column_stats": {
            col: {
                "mean": float(table[col].mean()) if not table.empty else math.nan,
                "std": float(table[col].std(ddof=0)) if not table.empty else math.nan,
                "na_count": int(table[col].isna().sum()) if not table.empty else 0,
            }
            for col in table.columns
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)


def make_prior_table(cfg_path: str | Path) -> PriorTable:
    """Create the prior table from docking logs."""

    cfg = _load_yaml(Path(cfg_path))
    obps = _load_obp_list(Path(cfg["obp_list"]))
    metrics = cfg["metrics"]["order"]
    glob_pattern = cfg["input"]["glob"]
    dfs = _load_logs(glob_pattern)
    agg = _aggregate_logs(dfs, obps)
    if not agg:
        print("[prior] No docking logs found; generating empty prior table")
        table = _build_empty_prior(obps, metrics)
    else:
        table = _fill_and_stack(agg, obps, metrics, fill=math.nan)
        method = cfg.get("scaling", {}).get("method", "zscore")
        fill = cfg.get("scaling", {}).get("fillna", "mean")
        eps = float(cfg.get("scaling", {}).get("eps", 1e-6))
        if method == "zscore":
            table = _zscore(table, eps=eps, fill=fill)
        elif method == "percentile":
            table = table.rank(pct=True)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        table = table.fillna(table.mean() if fill == "mean" else 0.0)
    out_csv = Path(cfg["output"]["table_csv"])
    out_json = Path(cfg["output"]["qc_json"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv)
    _write_qc(out_json, table, obps, metrics)
    return PriorTable(table=table, obps=list(obps), metrics=list(metrics))


def load_prior_table(csv_path: str | Path) -> PriorTable:
    """Load an existing prior table CSV."""

    path = Path(csv_path)
    if not path.exists():
        return PriorTable(table=pd.DataFrame(), obps=[], metrics=[])
    df = pd.read_csv(path, index_col=0)
    obps: List[str] = []
    metrics: List[str] = []
    for col in df.columns:
        if "__" in col:
            obp, metric = col.split("__", 1)
        else:
            obp, metric = col, "value"
        if obp not in obps:
            obps.append(obp)
        if metric not in metrics:
            metrics.append(metric)
    return PriorTable(table=df, obps=obps, metrics=metrics)


def build_label_to_prior(
    labels: Sequence[str],
    prior: PriorTable,
    voc_map_yaml: str | Path,
    default: str = "mean",
) -> np.ndarray:
    """Map class labels to prior vectors using the provided VOC map."""

    mapping = _load_voc_map(Path(voc_map_yaml))
    default_vec = (
        prior.table.mean(axis=0).to_numpy(dtype=np.float32)
        if default == "mean" and not prior.table.empty
        else np.zeros(prior.prior_dim, dtype=np.float32)
    )
    out = []
    for label in labels:
        target = mapping.get(label, label)
        out.append(prior.lookup(target, default=default_vec))
    return np.stack(out, axis=0) if out else np.zeros((0, prior.prior_dim), dtype=np.float32)


__all__ = [
    "PriorTable",
    "make_prior_table",
    "load_prior_table",
    "build_label_to_prior",
]
