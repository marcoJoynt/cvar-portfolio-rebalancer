"""
Pytest hooks and shared config. VERBOSE=1 enables rebalancer DEBUG logging during tests.
"""
import os

import pytest


def pytest_configure(config):
    """Turn on rebalancer logger when VERBOSE=1."""
    if os.environ.get("VERBOSE") == "1":
        import logging
        from rebalancer._logging import configure_logging
        configure_logging(level=logging.DEBUG)
