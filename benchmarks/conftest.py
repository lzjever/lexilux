"""Configuration for benchmark tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
