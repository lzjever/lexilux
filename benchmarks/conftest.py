"""Benchmarks configuration."""

import pytest


def pytest_configure(config):
    """Configure pytest for benchmarks."""
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
