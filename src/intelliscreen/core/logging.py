"""Structured and standardized logging for IntelliScreen."""

import logging
import sys


def setup_logger(
    name: str = "intelliscreen",
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure and return a structured logger instance."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a namespaced child logger under intelliscreen."""
    return logging.getLogger(f"intelliscreen.{name}")
