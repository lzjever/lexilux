"""
Provider-specific configurations for Lexilux.

This module provides reasoning configuration and other provider-specific
settings that are not available in the models.dev database.
"""

from lexilux.providers.base import ReasoningConfig
from lexilux.providers.registry import get_reasoning_config, PROVIDER_URL_PATTERNS

__all__ = [
    "ReasoningConfig",
    "get_reasoning_config",
    "PROVIDER_URL_PATTERNS",
]
