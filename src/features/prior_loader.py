"""Utilities for loading docking priors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_priors(path: str | Path) -> Dict[str, Any]:
    """Load a prior table from JSON."""
    return json.loads(Path(path).read_text() or "{}");
