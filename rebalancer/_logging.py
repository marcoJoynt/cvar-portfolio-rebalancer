"""
Logging for the rebalancer. Use get_logger(__name__) in each module.
"""
import logging
from typing import Optional

_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name (e.g. __name__)."""
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """Configure the rebalancer logger. Idempotent."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    format_string = format_string or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    logger = logging.getLogger("rebalancer")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    _LOG_CONFIGURED = True
