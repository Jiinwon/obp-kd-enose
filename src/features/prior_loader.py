"""Utilities for loading docking priors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # optional dependencies
    import torch
except Exception:  # pragma: no cover - optional
    torch = None

try:  # optional
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None


def load_priors(path: str | Path) -> Tuple[Any, List[str]]:
    """Load a prior table from JSON into a tensor-like array of shape [C, D]."""
    data: Dict[str, List[float]] = json.loads(Path(path).read_text() or "{}")
    classes = sorted(data.keys())
    matrix_list = [data[c] for c in classes]

    if torch is not None:  # prefer torch tensor
        matrix = torch.tensor(matrix_list, dtype=torch.float32)
    elif np is not None:
        matrix = np.array(matrix_list, dtype=float)
    else:
        # Fallback simple object exposing .shape
        class _Simple:
            def __init__(self, data: List[List[float]]) -> None:
                self.data = data
                self.shape = (len(data), len(data[0]) if data else 0)
        matrix = _Simple(matrix_list)
    return matrix, classes
