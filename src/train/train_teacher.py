"""CLI entry point for training the teacher model."""
from __future__ import annotations

import argparse

from .utils_train import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_teacher.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # Placeholder: training loop would go here
    print(f"Loaded teacher config with epochs={cfg.get('epochs', 0)}")


if __name__ == "__main__":  # pragma: no cover
    main()
