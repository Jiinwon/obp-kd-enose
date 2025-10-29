"""CLI entry-point for running knowledge distillation experiments."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig, load_experiment_config
from src.train.loop import build_loaders, train_val_loop


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg_path: str | Path) -> None:
    """Run an experiment given a configuration path."""

    cfg: ExperimentConfig = load_experiment_config(cfg_path)
    _set_seed(cfg.seed)
    (
        train_loader,
        val_loader,
        n_classes,
        class_names,
        in_channels,
        window,
        prior_dim,
    ) = build_loaders(cfg)
    train_val_loop(
        cfg,
        train_loader,
        val_loader,
        in_channels,
        window,
        n_classes,
        prior_dim=prior_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OBP-KD experiment")
    parser.add_argument("--cfg", required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    run(args.cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
