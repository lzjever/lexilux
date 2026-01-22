"""
Pytest configuration and shared fixtures for lexilux tests.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Disable proxies for tests to avoid proxy-related issues
# This ensures tests connect directly to endpoints
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

# Add project root to Python path so tests can import lexilux
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest  # noqa: E402


def load_test_config() -> Optional[dict[str, Any]]:
    """
    Load test endpoints configuration from test_endpoints.json.

    Returns:
        Configuration dict if file exists, None otherwise.
    """
    config_path = Path(__file__).parent / "test_endpoints.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    return None


@pytest.fixture(scope="session")
def test_config():
    """Fixture to provide test configuration."""
    return load_test_config()


@pytest.fixture(scope="session")
def has_real_api_config(test_config):
    """Fixture to check if real API config is available."""
    return test_config is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line(
        "markers", "skip_if_no_config: skip test if test_endpoints.json is not available"
    )
