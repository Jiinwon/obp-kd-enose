"""Realtime serial reading and inference."""
from __future__ import annotations

from typing import Iterable


def read_serial(port: str) -> Iterable[str]:
    """Placeholder generator that yields lines from a serial port."""
    # In real usage this would wrap ``pyserial``.
    yield from []
