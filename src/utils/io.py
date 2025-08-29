"""I/O utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV file into a DataFrame."""
    return pd.read_csv(path)
