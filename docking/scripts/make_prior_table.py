"""Generate a 56-D prior table from docking logs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.prior_pipeline import make_prior_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build docking prior table")
    parser.add_argument("--cfg", required=True, help="Path to prior_config.yaml")
    args = parser.parse_args()
    make_prior_table(args.cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
