from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_enose import BaseENoseDataset


class DynamicMixtureDataset(BaseENoseDataset):
    def __init__(
        self, *, registry: Dict[str, Any], split: str, root_override: Optional[Union[str, Path]] = None
    ) -> None:
        super().__init__(
            registry=registry,
            dataset_key="dynamic",
            split=split,
            root_override=root_override,
        )
