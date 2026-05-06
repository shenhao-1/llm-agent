"""Simple logger helper for early phases."""

from __future__ import annotations

import logging


def get_logger(name: str = "modular_rag", level: int = logging.INFO) -> logging.Logger:
    """Return a stderr logger with a stable format and without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger
