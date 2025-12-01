"""Train teacher and student MLPs with knowledge distillation on the UCI gas dataset."""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.datasets import BatchSplitConfig, load_feature_batches, split_by_batches
from src.features import fit_minmax, save_scaler
from src.models import StudentConfig, TeacherConfig, build_student, build_teacher
from src.train.distillation import (
    DistillationConfig,
    OptimizerConfig,
    build_loaders,
    evaluate_classifier,
    train_student,
    train_teacher,
)

LOGGER = logging.getLogger("kd_gsa")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with batch*.dat files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/distillation"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-epochs", type=int, default=150)
    parser.add_argument("--student-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--train-batches",
        type=int,
        nargs="*",
        default=tuple(range(1, 7)),
        help="Batch indices for training (default: 1-6)",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        nargs="*",
        default=(7,),
        help="Batch indices for validation",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        nargs="*",
        default=(8, 9, 10),
        help="Batch indices for testing",
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_datasets(args: argparse.Namespace) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    LOGGER.info("Loading LibSVM batches from %s", args.data_root)
    X, y, batch_ids = load_feature_batches(args.data_root)
    splits = BatchSplitConfig(train=args.train_batches, val=args.val_batches, test=args.test_batches)
    datasets = split_by_batches(X, y, batch_ids, splits)
    scaled = fit_minmax(datasets)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaled.scaler, output_dir / "scaler.joblib")
    return {"train": scaled.train, "val": scaled.val, "test": scaled.test}


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_seed(args.seed)
    datasets = prepare_datasets(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(datasets, args.batch_size)

    teacher_cfg = TeacherConfig(input_dim=datasets["train"][0].shape[1], num_classes=args.num_classes)
    teacher = build_teacher(teacher_cfg)
    teacher_history, teacher_state, teacher_best_val = train_teacher(
        teacher,
        loaders,
        OptimizerConfig(epochs=args.teacher_epochs, lr=args.teacher_lr, batch_size=args.batch_size),
        device=device,
    )
    torch.save(teacher_state, args.output_dir / "teacher.pt")
    teacher_val_loader = loaders.get("val")
    teacher_test_loader = loaders.get("test")
    metrics: Dict[str, Dict[str, float]] = {
        "teacher": {
            "best_val_acc": teacher_best_val,
        }
    }
    if teacher_val_loader is not None:
        val_loss, val_acc = evaluate_classifier(teacher, teacher_val_loader, device)
        metrics["teacher"].update({"val_loss": val_loss, "val_acc": val_acc})
    if teacher_test_loader is not None:
        test_loss, test_acc = evaluate_classifier(teacher, teacher_test_loader, device)
        metrics["teacher"].update({"test_loss": test_loss, "test_acc": test_acc})

    student_cfg = StudentConfig(input_dim=datasets["train"][0].shape[1], num_classes=args.num_classes)
    student = build_student(student_cfg)
    student_history, student_state, student_best_val = train_student(
        student,
        teacher,
        loaders,
        OptimizerConfig(epochs=args.student_epochs, lr=args.student_lr, batch_size=args.batch_size),
        DistillationConfig(alpha=args.alpha, temperature=args.temperature),
        device=device,
    )
    torch.save(student_state, args.output_dir / "student.pt")
    student_val_loader = loaders.get("val")
    student_test_loader = loaders.get("test")
    metrics["student"] = {"best_val_acc": student_best_val}
    if student_val_loader is not None:
        val_loss, val_acc = evaluate_classifier(student, student_val_loader, device)
        metrics["student"].update({"val_loss": val_loss, "val_acc": val_acc})
    if student_test_loader is not None:
        test_loss, test_acc = evaluate_classifier(student, student_test_loader, device)
        metrics["student"].update({"test_loss": test_loss, "test_acc": test_acc})

    torch.onnx.export(
        student.cpu(),
        torch.zeros(1, datasets["train"][0].shape[1], dtype=torch.float32),
        args.output_dir / "student.onnx",
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
    )
    student.to(device)

    history_path = args.output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher": teacher_history.records,
                "student": student_history.records,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved training artefacts to %s", args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
