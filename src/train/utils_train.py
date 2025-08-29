"""Training utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_seed(seed: int = 0) -> None:
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
