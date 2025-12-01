"""Training utilities for knowledge distillation on feature vectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses.kd import KDLoss


__all__ = [
    "OptimizerConfig",
    "DistillationConfig",
    "TrainingHistory",
    "build_loaders",
    "train_teacher",
    "train_student",
    "evaluate_classifier",
]


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyper-parameters shared by teacher and student training."""

    epochs: int
    lr: float
    batch_size: int


@dataclass(frozen=True)
class DistillationConfig:
    """Hyper-parameters for the distillation loss."""

    alpha: float = 0.5
    temperature: float = 2.0


@dataclass
class TrainingHistory:
    """Container tracking metrics across epochs."""

    records: List[Dict[str, float]] = field(default_factory=list)

    def append(self, epoch: int, split: str, loss: float, acc: float, **kwargs: float) -> None:
        row: Dict[str, float] = {"epoch": float(epoch), "loss": float(loss), "acc": float(acc)}
        row.update({k: float(v) for k, v in kwargs.items()})
        row["split_label"] = split
        self.records.append(row)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_loaders(
    scaled_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Construct dataloaders for the available splits."""

    loaders: Dict[str, DataLoader] = {}
    for split, (features, labels) in scaled_datasets.items():
        if features.size == 0:
            continue
        loaders[split] = _make_loader(features, labels, batch_size, shuffle=split == "train")
    return loaders


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    is_train = optimizer is not None
    model.train(is_train)
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * features.size(0)
        total_correct += (logits.argmax(dim=1) == labels).float().sum().item()
        total_samples += features.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def train_teacher(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    config: OptimizerConfig,
    *,
    device: torch.device,
) -> Tuple[TrainingHistory, Dict[str, Tensor], float]:
    """Train the teacher model and return its history and best state."""

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = TrainingHistory()
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_val_acc = 0.0
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    if train_loader is None or val_loader is None:
        raise KeyError("Both 'train' and 'val' loaders are required for teacher training")
    model.to(device)
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss, val_acc = _run_epoch(model, val_loader, optimizer=None, device=device)
        history.append(epoch, "train", train_loss, train_acc)
        history.append(epoch, "val", val_loss, val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history, best_state, best_val_acc


def train_student(
    student: nn.Module,
    teacher: nn.Module,
    loaders: Dict[str, DataLoader],
    config: OptimizerConfig,
    distill: DistillationConfig,
    *,
    device: torch.device,
) -> Tuple[TrainingHistory, Dict[str, Tensor], float]:
    """Train the student model using knowledge distillation."""

    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr)
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = KDLoss(temperature=distill.temperature, alpha=1.0)
    history = TrainingHistory()
    best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    best_val_acc = 0.0
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    if train_loader is None or val_loader is None:
        raise KeyError("Both 'train' and 'val' loaders are required for student training")
    teacher.eval().to(device)
    student.to(device)
    alpha = float(distill.alpha)
    for epoch in range(1, config.epochs + 1):
        student.train()
        total_loss = total_ce = total_kd = 0.0
        total_correct = 0.0
        total_samples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(features)
            student_logits = student(features)
            ce = ce_loss(student_logits, labels)
            kd = kd_loss(student_logits, teacher_logits)
            loss = alpha * ce + (1.0 - alpha) * kd
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            total_ce += ce.item() * features.size(0)
            total_kd += kd.item() * features.size(0)
            total_correct += (student_logits.argmax(dim=1) == labels).float().sum().item()
            total_samples += features.size(0)
        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        train_ce = total_ce / max(total_samples, 1)
        train_kd = total_kd / max(total_samples, 1)
        val_loss, val_acc = _run_epoch(student, val_loader, optimizer=None, device=device)
        history.append(epoch, "train", train_loss, train_acc, ce=train_ce, kd=train_kd)
        history.append(epoch, "val", val_loss, val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    student.load_state_dict(best_state)
    return history, best_state, best_val_acc


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate a classifier returning (loss, accuracy)."""

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = total_correct = 0.0
    total_samples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * features.size(0)
        total_correct += (logits.argmax(dim=1) == labels).float().sum().item()
        total_samples += features.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc

