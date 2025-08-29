"""Single entry point for non-HPC users."""
from __future__ import annotations

from pathlib import Path


def run_demo(config_path: str) -> None:
    """Load a model and run a trivial demo."""
    if not Path(config_path).exists():
        raise FileNotFoundError(config_path)
    print(f"Running demo with config {config_path}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/user_infer.yaml")
    run_demo(parser.parse_args().config)
