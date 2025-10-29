from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_enose import BaseENoseDataset


class TempModDataset(BaseENoseDataset):
    def __init__(
        self, *, registry: Dict[str, Any], split: str, root_override: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__(
            registry=registry,
            dataset_key="tempmod",
            split=split,
            root_override=root_override,
        )
