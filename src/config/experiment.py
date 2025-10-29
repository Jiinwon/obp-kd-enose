"""Dataclasses for experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class PreprocessPaths:
    """Paths to preprocessing YAML files."""

    common: str
    dataset: str


@dataclass
class DataConfig:
    """Data section of the experiment configuration."""

    registry: str
    dataset_key: str
    splits: Dict[str, str]
    preprocess: PreprocessPaths
    synthetic_ok: bool
    class_names: List[str]


@dataclass
class PriorConfig:
    """Prior table configuration."""

    table_csv: str
    voc_map: str
    default_strategy: str = "mean"


@dataclass
class ModelConfig:
    """Model hyperparameters."""

    type: str
    in_channels: int
    prior_dim: int
    hidden: int
    n_blocks: int
    kernel_size: int
    n_classes: int


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    amp: bool
    log_dir: str


@dataclass
class LossConfig:
    """Loss configuration."""

    lambda_kd: float


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str
    seed: int
    data: DataConfig
    prior: PriorConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse an experiment YAML file into :class:`ExperimentConfig`."""

    cfg = _load_yaml(Path(path))
    data_cfg = cfg["data"]
    preprocess_cfg = data_cfg.get("preprocess", {})
    preprocess = PreprocessPaths(
        common=str(preprocess_cfg.get("common")),
        dataset=str(preprocess_cfg.get("dataset")),
    )
    data = DataConfig(
        registry=str(data_cfg["registry"]),
        dataset_key=str(data_cfg["dataset_key"]),
        splits={k: str(v) for k, v in data_cfg.get("splits", {}).items()},
        preprocess=preprocess,
        synthetic_ok=bool(data_cfg.get("synthetic_ok", False)),
        class_names=[str(v) for v in data_cfg.get("class_names", [])],
    )
    prior_cfg = cfg["prior"]
    prior = PriorConfig(
        table_csv=str(prior_cfg["table_csv"]),
        voc_map=str(prior_cfg["voc_map"]),
        default_strategy=str(prior_cfg.get("default_strategy", "mean")),
    )
    model_cfg = cfg["model"]
    model = ModelConfig(
        type=str(model_cfg["type"]),
        in_channels=int(model_cfg["in_channels"]),
        prior_dim=int(model_cfg["prior_dim"]),
        hidden=int(model_cfg["hidden"]),
        n_blocks=int(model_cfg["n_blocks"]),
        kernel_size=int(model_cfg["kernel_size"]),
        n_classes=int(model_cfg["n_classes"]),
    )
    train_cfg = cfg["train"]
    train = TrainConfig(
        epochs=int(train_cfg["epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        amp=bool(train_cfg.get("amp", False)),
        log_dir=str(train_cfg["log_dir"]),
    )
    loss_cfg = cfg["loss"]
    loss = LossConfig(lambda_kd=float(loss_cfg["lambda_kd"]))
    return ExperimentConfig(
        experiment_name=str(cfg["experiment_name"]),
        seed=int(cfg.get("seed", 0)),
        data=data,
        prior=prior,
        model=model,
        train=train,
        loss=loss,
    )


__all__ = [
    "PreprocessPaths",
    "DataConfig",
    "PriorConfig",
    "ModelConfig",
    "TrainConfig",
    "LossConfig",
    "ExperimentConfig",
    "load_experiment_config",
]
