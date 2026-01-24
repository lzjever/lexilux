"""
Tests for the Model Registry and ChatFactory.
"""

import json
import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from lexilux.registry import (
    ChatFactory,
    ConfiguredChat,
    ModelCapabilities,
    ModelLimits,
    ModelModalities,
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)


# === Test Data ===

SAMPLE_REGISTRY_DATA = {
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "api": "https://api.openai.com/v1",
        "doc": "https://platform.openai.com/docs",
        "env": ["OPENAI_API_KEY"],
        "npm": "@ai-sdk/openai",
        "models": {
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "family": "gpt",
                "attachment": True,
                "reasoning": False,
                "tool_call": True,
                "structured_output": True,
                "temperature": True,
                "knowledge": "2023-12",
                "release_date": "2024-05-13",
                "last_updated": "2024-05-13",
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "open_weights": False,
                "cost": {"input": 2.5, "output": 10.0},
                "limit": {"context": 128000, "output": 16384},
            },
            "gpt-3.5-turbo": {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "family": "gpt",
                "attachment": False,
                "reasoning": False,
                "tool_call": True,
                "temperature": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "open_weights": False,
                "cost": {"input": 0.5, "output": 1.5},
                "limit": {"context": 16385, "output": 4096},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "api": "https://api.anthropic.com/v1",
        "env": ["ANTHROPIC_API_KEY"],
        "models": {
            "claude-3-opus": {
                "id": "claude-3-opus",
                "name": "Claude 3 Opus",
                "family": "claude-opus",
                "reasoning": True,
                "tool_call": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 200000, "output": 4096},
            },
        },
    },
    "groq": {
        "id": "groq",
        "name": "Groq",
        "api": "https://api.groq.com/openai/v1",
        "env": ["GROQ_API_KEY"],
        "models": {
            "llama-3.1-70b": {
                "id": "llama-3.1-70b",
                "name": "Llama 3.1 70B",
                "family": "llama",
                "reasoning": False,
                "tool_call": True,
                "open_weights": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 131072, "output": 8192},
            },
        },
    },
}


@pytest.fixture
def sample_data_file():
    """Create a temporary file with sample registry data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_REGISTRY_DATA, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def registry(sample_data_file):
    """Create a registry with sample data."""
    # Reset singleton before each test
    ModelRegistry.reset_instance()
    return ModelRegistry(data_path=sample_data_file)


@pytest.fixture
def factory(registry):
    """Create a factory with the sample registry."""
    return ChatFactory(registry=registry)


# === ModelSpec Tests ===


class TestModelSpec:
    """Tests for ModelSpec data class."""

    def test_default_values(self):
        """Test ModelSpec with default values."""
        spec = ModelSpec(id="test", name="Test Model", provider_id="test-provider")

        assert spec.id == "test"
        assert spec.name == "Test Model"
        assert spec.provider_id == "test-provider"
        assert spec.capabilities.tool_call is False
        assert spec.limits.context == 8192
        assert spec.modalities.input == ("text",)

    def test_with_capabilities(self):
        """Test ModelSpec with custom capabilities."""
        caps = ModelCapabilities(tool_call=True, reasoning=True)
        spec = ModelSpec(
            id="test",
            name="Test",
            provider_id="test",
            capabilities=caps,
        )

        assert spec.supports_tool_call is True
        assert spec.supports_reasoning is True

    def test_convenience_properties(self):
        """Test convenience properties."""
        spec = ModelSpec(
            id="test",
            name="Test",
            provider_id="test",
            capabilities=ModelCapabilities(tool_call=True),
            limits=ModelLimits(context=128000, output=16384),
            modalities=ModelModalities(input=("text", "image"), output=("text",)),
        )

        assert spec.supports_vision is True
        assert spec.supports_audio is False
        assert spec.context_window == 128000
        assert spec.max_output == 16384


class TestProviderSpec:
    """Tests for ProviderSpec data class."""

    def test_provider_spec(self):
        """Test ProviderSpec creation."""
        provider = ProviderSpec(
            id="openai",
            name="OpenAI",
            api_base="https://api.openai.com/v1",
            env_vars=("OPENAI_API_KEY",),
        )

        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        assert provider.api_base == "https://api.openai.com/v1"
        assert "OPENAI_API_KEY" in provider.env_vars


# === ModelRegistry Tests ===


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_load_from_file(self, sample_data_file):
        """Test loading registry from file."""
        registry = ModelRegistry(data_path=sample_data_file)

        assert registry.is_loaded is True
        assert registry.provider_count() == 3
        assert registry.count() == 4

    def test_providers(self, registry):
        """Test iterating providers."""
        providers = list(registry.providers())

        assert len(providers) == 3
        provider_ids = [p.id for p in providers]
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids

    def test_get_provider(self, registry):
        """Test getting a specific provider."""
        provider = registry.get_provider("openai")

        assert provider is not None
        assert provider.name == "OpenAI"
        assert provider.api_base == "https://api.openai.com/v1"
        assert "OPENAI_API_KEY" in provider.env_vars

    def test_get_provider_not_found(self, registry):
        """Test getting non-existent provider."""
        provider = registry.get_provider("nonexistent")
        assert provider is None

    def test_models_all(self, registry):
        """Test iterating all models."""
        models = list(registry.models())
        assert len(models) == 4

    def test_models_by_provider(self, registry):
        """Test iterating models by provider."""
        models = list(registry.models(provider="openai"))

        assert len(models) == 2
        model_ids = [m.id for m in models]
        assert "gpt-4o" in model_ids
        assert "gpt-3.5-turbo" in model_ids

    def test_get_model(self, registry):
        """Test getting a specific model."""
        spec = registry.get("gpt-4o", provider="openai")

        assert spec.id == "gpt-4o"
        assert spec.name == "GPT-4o"
        assert spec.family == "gpt"
        assert spec.capabilities.tool_call is True
        assert spec.limits.context == 128000
        assert "image" in spec.modalities.input

    def test_get_model_without_provider(self, registry):
        """Test getting model without specifying provider."""
        spec = registry.get("gpt-4o")

        assert spec.id == "gpt-4o"
        assert spec.provider_id == "openai"

    def test_get_unknown_model(self, registry, caplog):
        """Test getting unknown model returns conservative defaults."""
        with caplog.at_level(logging.WARNING):
            spec = registry.get("unknown-model")

        assert spec.id == "unknown-model"
        assert spec.limits.context == 8192
        assert spec.limits.output == 4096
        assert spec.capabilities.tool_call is False
        assert "not found in registry" in caplog.text

    def test_get_unknown_model_suppress_warning(self, registry, caplog):
        """Test suppressing warning for unknown model."""
        with caplog.at_level(logging.WARNING):
            spec = registry.get("unknown-model", suppress_unknown_warning=True)

        assert spec.id == "unknown-model"
        assert "not found in registry" not in caplog.text

    def test_search_by_capability(self, registry):
        """Test searching models by capability."""
        # Find reasoning models
        reasoning_models = list(registry.search(supports_reasoning=True))

        assert len(reasoning_models) == 1
        assert reasoning_models[0].id == "claude-3-opus"

    def test_search_by_vision(self, registry):
        """Test searching for vision models."""
        vision_models = list(registry.search(supports_vision=True))

        assert len(vision_models) == 2
        model_ids = [m.id for m in vision_models]
        assert "gpt-4o" in model_ids
        assert "claude-3-opus" in model_ids

    def test_search_by_min_context(self, registry):
        """Test searching by minimum context."""
        large_context = list(registry.search(min_context=100000))

        assert len(large_context) == 3
        for model in large_context:
            assert model.limits.context >= 100000

    def test_search_by_open_weights(self, registry):
        """Test searching for open weight models."""
        open_models = list(registry.search(open_weights=True))

        assert len(open_models) == 1
        assert open_models[0].id == "llama-3.1-70b"

    def test_search_combined_criteria(self, registry):
        """Test searching with multiple criteria."""
        results = list(
            registry.search(
                supports_tool_call=True, supports_vision=True, min_context=100000
            )
        )

        assert len(results) == 2
        model_ids = [m.id for m in results]
        assert "gpt-4o" in model_ids
        assert "claude-3-opus" in model_ids

    def test_get_providers_for_model(self, registry):
        """Test getting providers that offer a model."""
        providers = registry.get_providers_for_model("gpt-4o")

        assert len(providers) == 1
        assert "openai" in providers

    def test_singleton(self, sample_data_file):
        """Test singleton behavior."""
        ModelRegistry.reset_instance()

        # Patch the data loading to use our sample data
        with patch.object(ModelRegistry, "_load_data") as mock_load:
            mock_load.side_effect = lambda path: None

            instance1 = ModelRegistry.get_instance()
            instance2 = ModelRegistry.get_instance()

            assert instance1 is instance2

        ModelRegistry.reset_instance()


# === ChatFactory Tests ===


class TestChatFactory:
    """Tests for ChatFactory."""

    def test_create_with_explicit_config(self, factory):
        """Test creating chat with explicit configuration."""
        chat = factory.create(
            "gpt-4o",
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

        assert isinstance(chat, ConfiguredChat)
        assert chat.model == "gpt-4o"
        assert chat.model_spec.id == "gpt-4o"
        assert chat.supports_tool_call is True
        assert chat.context_limit == 128000

    def test_create_auto_resolve_base_url(self, factory):
        """Test auto-resolving base URL from registry."""
        chat = factory.create("gpt-4o", provider="openai", api_key="test-key")

        assert chat.base_url == "https://api.openai.com/v1"

    def test_create_auto_resolve_api_key(self, factory, monkeypatch):
        """Test auto-resolving API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

        chat = factory.create("gpt-4o", provider="openai")

        assert chat.api_key == "env-test-key"

    def test_create_unknown_model(self, factory, caplog):
        """Test creating chat with unknown model."""
        with caplog.at_level(logging.WARNING):
            chat = factory.create(
                "unknown-model",
                base_url="http://localhost:8080/v1",
            )

        assert chat.model == "unknown-model"
        assert chat.context_limit == 8192
        assert "not found in registry" in caplog.text

    def test_create_unknown_model_suppress_warning(self, factory, caplog):
        """Test suppressing warning for unknown model."""
        with caplog.at_level(logging.WARNING):
            chat = factory.create(
                "unknown-model",
                base_url="http://localhost:8080/v1",
                suppress_unknown_warning=True,
            )

        assert chat.model == "unknown-model"
        assert "not found in registry" not in caplog.text

    def test_create_missing_base_url(self, factory):
        """Test error when base_url cannot be resolved."""
        with pytest.raises(ValueError, match="base_url is required"):
            factory.create("unknown-model")

    def test_list_providers(self, factory):
        """Test listing providers."""
        providers = factory.list_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) == 3

    def test_list_models(self, factory):
        """Test listing models."""
        all_models = factory.list_models()
        assert len(all_models) == 4

        openai_models = factory.list_models(provider="openai")
        assert len(openai_models) == 2
        assert "gpt-4o" in openai_models

    def test_search_models(self, factory):
        """Test searching models through factory."""
        results = list(factory.search_models(supports_reasoning=True))

        assert len(results) == 1
        assert results[0].id == "claude-3-opus"


# === ConfiguredChat Tests ===


class TestConfiguredChat:
    """Tests for ConfiguredChat."""

    def test_properties(self, factory):
        """Test ConfiguredChat properties."""
        chat = factory.create("gpt-4o", provider="openai", api_key="test-key")

        assert chat.supports_tool_call is True
        assert chat.supports_reasoning is False
        assert chat.supports_vision is True
        assert chat.supports_audio is False
        assert chat.context_limit == 128000
        assert chat.output_limit == 16384
        assert chat.model_family == "gpt"

    def test_get_recommended_params(self, factory):
        """Test getting recommended parameters."""
        chat = factory.create("gpt-4o", provider="openai", api_key="test-key")

        params = chat.get_recommended_params()

        assert params.temperature == 0.7
        assert params.max_tokens <= chat.output_limit
        assert params.max_tokens > 0

    def test_get_recommended_params_custom_ratio(self, factory):
        """Test recommended params with custom ratio."""
        chat = factory.create("gpt-4o", provider="openai", api_key="test-key")

        params = chat.get_recommended_params(max_tokens_ratio=0.25)

        assert params.max_tokens == int(16384 * 0.25)

    def test_repr(self, factory):
        """Test string representation."""
        chat = factory.create("gpt-4o", provider="openai", api_key="test-key")

        repr_str = repr(chat)

        assert "ConfiguredChat" in repr_str
        assert "gpt-4o" in repr_str
        assert "128000" in repr_str


# === Integration Tests ===


class TestRegistryIntegration:
    """Integration tests using real bundled data."""

    def test_load_bundled_data(self):
        """Test loading the bundled models.json."""
        ModelRegistry.reset_instance()

        try:
            registry = ModelRegistry()

            # Should have loaded data
            assert registry.is_loaded is True
            assert registry.provider_count() > 0
            assert registry.count() > 0

            # Should have common providers
            openai = registry.get_provider("openai")
            if openai:
                assert openai.name == "OpenAI"

        finally:
            ModelRegistry.reset_instance()

    def test_factory_with_bundled_data(self):
        """Test factory with bundled data."""
        ModelRegistry.reset_instance()

        try:
            factory = ChatFactory()

            # Should be able to list providers
            providers = factory.list_providers()
            assert len(providers) > 0

        finally:
            ModelRegistry.reset_instance()
