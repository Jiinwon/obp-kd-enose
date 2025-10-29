"""Temperature modulation dataset wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_enose import BaseENoseDataset, load_data_registry


class TempModDataset(BaseENoseDataset):
    def __init__(
        self,
        *,
        split: str,
        registry: Optional[Dict[str, Any]] = None,
        registry_path: str = "configs/data_registry.yaml",
        root_override: Optional[Union[str, Path]] = None,
        synthetic_ok: bool = False,
    ) -> None:
        if registry is None:
            registry = load_data_registry(registry_path)
        super().__init__(
            registry=registry,
            dataset_key="tempmod",
            split=split,
            root_override=root_override,
            synthetic_ok=synthetic_ok,
        )


__all__ = ["TempModDataset"]
