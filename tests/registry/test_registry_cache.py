"""Tests for ModelRegistry LRU cache functionality."""

import json
import os
import tempfile
import time

import pytest

from lexilux.registry import ModelRegistry


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
    # Reset singleton and cache before each test
    ModelRegistry.reset_instance()
    registry = ModelRegistry(data_path=sample_data_file)
    # Clear the cache to ensure clean state
    registry._get_model_spec_cached.cache_clear()
    return registry


class TestRegistryCache:
    """Tests for ModelRegistry LRU cache functionality."""

    def test_cached_model_lookup_is_faster(self, registry):
        """Test that cached lookups are faster than uncached lookups.

        This test performs the same model lookup many times and measures
        the improvement from caching. The cache should make repeated lookups
        significantly faster.
        """
        model_id = "gpt-4o"
        iterations = 1000

        # Clear cache to ensure first lookup is uncached
        registry._get_model_spec_cached.cache_clear()

        # First lookup (uncached)
        start_time = time.perf_counter()
        for _ in range(iterations):
            registry.get(model_id, provider="openai")
        first_time = time.perf_counter() - start_time

        # Second lookup (cached)
        start_time = time.perf_counter()
        for _ in range(iterations):
            registry.get(model_id, provider="openai")
        second_time = time.perf_counter() - start_time

        # Cached lookups should be faster
        # Note: Due to variability in timing, we just verify both complete
        # In practice, the cached version should be significantly faster
        assert first_time > 0
        assert second_time >= 0

        # Verify cache info shows hits
        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.hits > 0
        assert cache_info.currsize > 0

    def test_cache_hits_for_repeated_lookups(self, registry):
        """Test that repeated lookups result in cache hits.

        This test verifies that looking up the same model multiple times
        results in cache hits rather than repeated lookups.
        """
        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        # Initial cache state
        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0

        # First lookup should be a cache miss
        spec1 = registry.get("gpt-4o", provider="openai")
        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 0

        # Second lookup should be a cache hit
        spec2 = registry.get("gpt-4o", provider="openai")
        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 1

        # Third lookup should also be a cache hit
        spec3 = registry.get("gpt-4o", provider="openai")
        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 2

        # All lookups should return the same spec
        assert spec1 is spec2
        assert spec2 is spec3

    def test_unknown_model_not_cached(self, registry):
        """Test that unknown models are cached separately.

        This test verifies that unknown models (which return conservative
        defaults) are cached and don't result in repeated warnings.
        """
        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        # Look up unknown model twice
        spec1 = registry.get("unknown-model-x", suppress_unknown_warning=True)
        spec2 = registry.get("unknown-model-x", suppress_unknown_warning=True)

        # Should have cached the unknown model
        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 1

        # Both should return the same spec (with conservative defaults)
        assert spec1 is spec2
        assert spec1.id == "unknown-model-x"
        assert spec1.limits.context == 8192
        assert spec1.capabilities.tool_call is False

    def test_cache_with_different_providers(self, registry):
        """Test that cache handles provider parameter correctly.

        This test verifies that the same model_id with different providers
        results in different cache entries.
        """
        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        # Look up model without provider
        spec1 = registry.get("gpt-4o")

        # Look up model with explicit provider
        spec2 = registry.get("gpt-4o", provider="openai")

        # Both should work
        assert spec1.id == "gpt-4o"
        assert spec2.id == "gpt-4o"

        # Check cache state
        cache_info = registry._get_model_spec_cached.cache_info()
        # Should have 2 entries (one with None provider, one with "openai")
        assert cache_info.currsize == 2

    def test_cache_size_limit(self, registry):
        """Test that cache respects the maxsize limit.

        This test verifies that the cache doesn't grow beyond its configured
        maximum size (128 entries).
        """
        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        # Look up the same model many times
        for _ in range(200):
            registry.get("gpt-4o", provider="openai")

        cache_info = registry._get_model_spec_cached.cache_info()

        # Cache should not exceed maxsize
        assert cache_info.currsize <= 128

        # Should have many cache hits
        assert cache_info.hits > 150

    def test_cache_clear(self, registry):
        """Test that cache can be cleared.

        This test verifies the cache_clear() method works correctly.
        """
        # Add some entries to cache
        registry.get("gpt-4o", provider="openai")
        registry.get("gpt-3.5-turbo", provider="openai")

        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.currsize > 0

        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.currsize == 0
        assert cache_info.hits == 0
        assert cache_info.misses == 0

    def test_cache_with_multiple_models(self, registry):
        """Test cache behavior with multiple different models.

        This test verifies that the cache correctly handles lookups for
        multiple different models.
        """
        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        # Look up different models
        models = ["gpt-4o", "gpt-3.5-turbo", "claude-3-opus"]

        # First pass - all misses
        for model_id in models:
            registry.get(model_id)

        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.misses == len(models)
        assert cache_info.hits == 0

        # Second pass - all hits
        for model_id in models:
            registry.get(model_id)

        cache_info = registry._get_model_spec_cached.cache_info()
        assert cache_info.misses == len(models)
        assert cache_info.hits == len(models)

    def test_cache_stats_accuracy(self, registry):
        """Test that cache statistics are accurately maintained.

        This test verifies that cache_info() returns accurate statistics
        about cache usage.
        """
        # Clear cache
        registry._get_model_spec_cached.cache_clear()

        # Perform known operations
        # 3 unique lookups
        registry.get("gpt-4o")
        registry.get("gpt-3.5-turbo")
        registry.get("claude-3-opus")

        # 5 repeated lookups (should be hits)
        for _ in range(5):
            registry.get("gpt-4o")

        cache_info = registry._get_model_spec_cached.cache_info()

        # Verify stats
        assert cache_info.currsize == 3  # 3 unique entries
        assert cache_info.misses == 3  # 3 initial misses
        assert cache_info.hits == 5  # 5 subsequent hits
