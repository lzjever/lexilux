"""
Pytest configuration and shared fixtures for lexilux tests.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to Python path so tests can import lexilux
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest  # noqa: E402


def load_test_config() -> Optional[Dict[str, Any]]:
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
        except (json.JSONDecodeError, IOError):
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
    config.addinivalue_line("markers", "real_api: mark test as requiring real API endpoints")
    config.addinivalue_line(
        "markers", "skip_if_no_config: skip test if test_endpoints.json is not available"
    )
