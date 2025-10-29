"""Training loop utilities for the MVP pipeline."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.datasets import BaseENoseDataset, build_dataset
from src.features.prior_pipeline import build_label_to_prior, load_prior_table
from src.features.preprocessing import PreprocessPipeline
from src.models.student import StudentModel
from src.utils.logger import get_logger


@dataclass
class Metrics:
    records: List[Dict[str, float]]

    def append(self, record: Dict[str, float]) -> None:
        self.records.append(record)

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "acc", "ce", "kd"])
            writer.writeheader()
            for row in self.records:
                writer.writerow(row)


class WindowedENoseDataset(Dataset):
    """Dataset providing preprocessed windows."""

    def __init__(
        self,
        base: BaseENoseDataset,
        pipeline: PreprocessPipeline,
        label_to_index: Dict[str, int],
        prior_vectors: np.ndarray,
    ) -> None:
        self.samples: List[Tuple[np.ndarray, int, np.ndarray]] = []
        for X, y, meta in base:
            aux = {"temp": meta.temp, "humid": meta.humid}
            windows = pipeline.transform_and_window(X, device_id=meta.device_id, aux=aux)
            label = y.get("gas") or list(label_to_index.keys())[0]
            if label not in label_to_index:
                continue
            label_idx = label_to_index[label]
            prior_vec = prior_vectors[label_idx]
            for window in windows:
                self.samples.append((window.astype(np.float32), label_idx, prior_vec.astype(np.float32)))
        if not self.samples:
            raise RuntimeError("No windows generated; check preprocessing configuration")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        window, label, prior = self.samples[idx]
        return (
            torch.from_numpy(window),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(prior),
        )


def _fit_pipeline(pipeline: PreprocessPipeline, dataset: BaseENoseDataset) -> None:
    train_samples = [dataset[i] for i in range(len(dataset))]
    device_ids = [meta.device_id for (_, _, meta) in train_samples]
    aux_provider = lambda meta: {"temp": meta.temp, "humid": meta.humid}
    pipeline.fit(train_samples, train_device_ids=device_ids, aux_provider=aux_provider)


def _build_windowed_dataset(
    base: BaseENoseDataset,
    pipeline: PreprocessPipeline,
    label_to_index: Dict[str, int],
    prior_vectors: np.ndarray,
) -> WindowedENoseDataset:
    return WindowedENoseDataset(base, pipeline, label_to_index, prior_vectors)


def build_loaders(cfg: ExperimentConfig):
    """Create dataloaders for training and validation."""

    train_base = build_dataset(
        cfg.data.dataset_key,
        cfg.data.splits["train"],
        cfg.data.registry,
        synthetic_ok=cfg.data.synthetic_ok,
    )
    val_base = build_dataset(
        cfg.data.dataset_key,
        cfg.data.splits["val"],
        cfg.data.registry,
        synthetic_ok=cfg.data.synthetic_ok,
    )
    pipeline = PreprocessPipeline.from_yaml(cfg.data.preprocess.common, cfg.data.preprocess.dataset)
    _fit_pipeline(pipeline, train_base)
    prior_table = load_prior_table(cfg.prior.table_csv)
    prior_vectors = build_label_to_prior(
        cfg.data.class_names, prior_table, cfg.prior.voc_map, cfg.prior.default_strategy
    )
    if prior_vectors.size == 0:
        prior_vectors = np.zeros((len(cfg.data.class_names), cfg.model.prior_dim), dtype=np.float32)
    label_to_index = {label: idx for idx, label in enumerate(cfg.data.class_names)}
    train_dataset = _build_windowed_dataset(train_base, pipeline, label_to_index, prior_vectors)
    val_dataset = _build_windowed_dataset(val_base, pipeline, label_to_index, prior_vectors)
    window = pipeline.cfg.window["size"]
    in_channels = train_dataset[0][0].shape[0]
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    prior_dim = prior_vectors.shape[1] if prior_vectors.size else cfg.model.prior_dim
    return train_loader, val_loader, len(cfg.data.class_names), cfg.data.class_names, in_channels, window, prior_dim


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    lambda_kd: float,
    amp: bool,
) -> Tuple[float, float, float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    total_loss = total_ce = total_kd = 0.0
    total_correct = 0.0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for windows, labels, priors in loader:
        windows = windows.to(device)
        labels = labels.to(device)
        priors = priors.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            logits, z_hat = model(windows, prior=priors, return_aux=True)
            ce = ce_loss(logits, labels)
            kd = mse_loss(z_hat, priors)
            loss = ce + lambda_kd * kd
        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * labels.size(0)
        total_ce += ce.item() * labels.size(0)
        total_kd += kd.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).float().sum().item()
        total_samples += labels.size(0)
    avg_loss = total_loss / max(1, total_samples)
    avg_ce = total_ce / max(1, total_samples)
    avg_kd = total_kd / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, avg_ce, avg_kd, acc


def train_val_loop(
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_channels: int,
    window: int,
    n_classes: int,
    *,
    prior_dim: int,
) -> Metrics:
    """Run the training/validation loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentModel(
        num_classes=n_classes,
        hidden_dim=cfg.model.hidden,
        in_channels=in_channels,
        use_film=cfg.model.type == "film_tcn",
        prior_dim=prior_dim,
        film_hidden=cfg.model.hidden,
        film_blocks=cfg.model.n_blocks,
        film_kernel=cfg.model.kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    logger = get_logger(__name__)
    metrics = Metrics(records=[])
    metrics_path = Path(cfg.train.log_dir) / "metrics.csv"
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_ce, train_kd, train_acc = _run_epoch(
            model, train_loader, optimizer, device, cfg.loss.lambda_kd, cfg.train.amp
        )
        val_loss, val_ce, val_kd, val_acc = _run_epoch(
            model, val_loader, None, device, cfg.loss.lambda_kd, cfg.train.amp
        )
        logger.info(
            "Epoch %d train loss=%.4f acc=%.3f kd=%.4f | val loss=%.4f acc=%.3f",
            epoch,
            train_loss,
            train_acc,
            train_kd,
            val_loss,
            val_acc,
        )
        metrics.append({"epoch": epoch, "split": "train", "loss": train_loss, "acc": train_acc, "ce": train_ce, "kd": train_kd})
        metrics.append({"epoch": epoch, "split": "val", "loss": val_loss, "acc": val_acc, "ce": val_ce, "kd": val_kd})
    metrics.to_csv(metrics_path)
    return metrics


__all__ = ["build_loaders", "train_val_loop", "Metrics"]
