"""Logging factory for SkyWatch AI.

Provides a consistent logger setup across all modules.
Each module gets its own named logger for per-module level control.
"""

import logging

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name, typically the module's ``__name__``.
        level: Logging level as a string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)

    return logger
