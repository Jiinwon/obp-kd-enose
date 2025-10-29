"""Simple logging helper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str, *, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger configured for console output.

    Parameters
    ----------
    name:
        Logger name.
    log_file:
        Optional path where logs should also be written.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


__all__ = ["get_logger"]
