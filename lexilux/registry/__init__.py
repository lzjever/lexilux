"""
Model Registry module.

Provides model specifications, provider information, and factory classes
for creating pre-configured Chat clients.

Usage:
    >>> from lexilux.registry import ChatFactory, ModelRegistry
    >>>
    >>> # Create a factory
    >>> factory = ChatFactory()
    >>>
    >>> # Create a pre-configured chat client
    >>> chat = factory.create("gpt-4o", provider="openai", api_key="...")
    >>>
    >>> # Check model capabilities
    >>> print(chat.supports_tool_call)  # True
    >>> print(chat.context_limit)       # 128000
    >>>
    >>> # Query the registry directly
    >>> registry = ModelRegistry.get_instance()
    >>> for model in registry.search(supports_reasoning=True):
    ...     print(model.name)
"""

from lexilux.registry.factory import ChatFactory, ConfiguredChat
from lexilux.registry.models import (
    ModelCapabilities,
    ModelCost,
    ModelLimits,
    ModelModalities,
    ModelSpec,
    ProviderSpec,
)
from lexilux.registry.registry import ModelRegistry

__all__ = [
    # Main classes
    "ModelRegistry",
    "ChatFactory",
    "ConfiguredChat",
    # Data classes
    "ModelSpec",
    "ProviderSpec",
    "ModelCapabilities",
    "ModelLimits",
    "ModelModalities",
    "ModelCost",
]
