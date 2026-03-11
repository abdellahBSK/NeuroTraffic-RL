"""Structured logging setup for NeuroTraffic-RL.

All modules obtain their logger via ``get_logger(__name__)``.
The root log level is read from the ``LOG_LEVEL`` environment variable
(default: INFO). Log files are written to ``logs/neurotraffic.log``.
"""

import logging
import os
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with console and file handlers.

    Call once per module at import time::

        from utils.logger import get_logger
        logger = get_logger(__name__)

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance ready to use.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when module is re-imported.
    if logger.handlers:
        return logger

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path(os.getenv("LOGS_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "neurotraffic.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger.
    logger.propagate = False

    return logger
