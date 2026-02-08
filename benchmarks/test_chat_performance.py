"""Chat performance benchmarks.

This module contains benchmarks for measuring the performance of chat operations
including latency, payload building, and model registry lookups.
"""

from unittest import mock

import pytest

from lexilux import Chat
from lexilux.chat._request import build_api_messages, build_params_dict, build_payload
from lexilux.chat.params import ChatParams
from lexilux.registry.registry import ModelRegistry


@pytest.mark.benchmark
def test_chat_latency_single_request(benchmark):
    """Benchmark latency of single chat request.

    This test measures the time it takes to prepare a single chat request
    from message normalization through payload building.

    The benchmark simulates the local preparation work that happens before
    any network request is made, providing a baseline for the client-side
    overhead of chat operations.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Setup test data
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    model = "gpt-4o"
    system = "You are a helpful assistant."

    # Benchmark the request preparation pipeline
    def prepare_single_request():
        # Step 1: Normalize messages
        api_messages = build_api_messages(
            messages=messages,
            system=system,
            history=None,
        )

        # Step 2: Build parameters
        params = build_params_dict(
            params=None,
            temperature=0.7,
            top_p=None,
            max_tokens=None,
            stop=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            user=None,
            n=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
        )

        # Step 3: Build final payload
        payload = build_payload(
            model=model,
            messages=api_messages,
            params=params,
            stream=False,
            include_usage=False,
            extra=None,
        )

        return payload

    result = benchmark(prepare_single_request)

    # Verify the result is valid
    assert result is not None
    assert "model" in result
    assert "messages" in result
    assert result["model"] == model
    assert len(result["messages"]) == 2  # system + user message


@pytest.mark.benchmark
def test_chat_payload_building(benchmark):
    """Benchmark payload building performance.

    This test measures the performance of building chat request payloads
    with different levels of complexity. It tests the build_payload function
    which is a critical path in every chat request.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Prepare test data
    model = "gpt-4o"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about it."},
    ]
    params = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    }

    # Benchmark payload building
    result = benchmark(
        build_payload,
        model=model,
        messages=messages,
        params=params,
        stream=False,
        include_usage=True,
        extra=None,
    )

    # Verify result
    assert result is not None
    assert result["model"] == model
    assert len(result["messages"]) == 4
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 1000
    # Note: stream key and stream_options are only added when stream=True
    # For stream=False, these keys are not included in the payload


@pytest.mark.benchmark
def test_model_registry_lookup(benchmark):
    """Benchmark ModelRegistry lookup performance.

    This test measures the performance of looking up model specifications
    from the ModelRegistry. This is an important operation for model
    capability checks and provider queries.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Get registry instance
    registry = ModelRegistry.get_instance()

    # Test model IDs for lookup
    test_models = ["gpt-4o", "gpt-4-turbo", "claude-3-opus", "llama-3.1-70b"]

    # Benchmark model lookup
    def lookup_model():
        # Look up each model in sequence
        results = []
        for model_id in test_models:
            spec = registry.get(model_id, suppress_unknown_warning=True)
            results.append(spec)
        return results

    results = benchmark(lookup_model)

    # Verify results
    assert results is not None
    assert len(results) == len(test_models)
    for spec in results:
        assert spec is not None
        assert hasattr(spec, "id")
        assert hasattr(spec, "name")
        assert hasattr(spec, "capabilities")


@pytest.mark.benchmark
def test_chat_client_initialization(benchmark):
    """Benchmark Chat client initialization time.

    This test measures how long it takes to create a new Chat client instance,
    which includes setting up the HTTP session with connection pooling.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Benchmark client creation
    result = benchmark(
        Chat,
        base_url="https://api.example.com",
        api_key="test_key",
        model="gpt-4o",
    )

    # Verify client was created
    assert result is not None
    assert result.base_url == "https://api.example.com"
    assert hasattr(result, "_session")
    assert result._session is not None


@pytest.mark.benchmark
def test_chat_params_to_dict(benchmark):
    """Benchmark ChatParams.to_dict() performance.

    This test measures the performance of converting ChatParams to a dictionary,
    which is a common operation when preparing API requests.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Create ChatParams with various settings
    params = ChatParams(
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        stop=["END", "STOP"],
    )

    # Benchmark to_dict conversion
    result = benchmark(params.to_dict, exclude_none=True)

    # Verify result
    assert result is not None
    assert isinstance(result, dict)
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 1000
    assert result["top_p"] == 0.9
    assert result["stop"] == ["END", "STOP"]


@pytest.mark.benchmark
def test_message_normalization(benchmark):
    """Benchmark message normalization performance.

    This test measures the performance of normalizing different message formats
    into the standard API format.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    from lexilux.chat.utils import normalize_messages

    # Test different message formats
    simple_message = "Hello, world!"
    dict_message = {"role": "user", "content": "Hello, world!"}
    list_of_messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
    ]

    # Benchmark simple string normalization
    simple_result = benchmark(normalize_messages, simple_message, system=None)
    assert len(simple_result) == 1
    assert simple_result[0]["role"] == "user"


@pytest.mark.benchmark
def test_registry_provider_search(benchmark):
    """Benchmark provider search performance.

    This test measures the performance of searching for models by provider,
    which is a common operation when working with the ModelRegistry.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    registry = ModelRegistry.get_instance()

    # Benchmark getting models from a provider
    def get_provider_models():
        return list(registry.models(provider="openai"))

    results = benchmark(get_provider_models)

    # Verify results
    assert results is not None
    assert isinstance(results, list)
    # All results should be ModelSpec instances
    for model in results:
        assert hasattr(model, "id")
        assert hasattr(model, "name")
        assert model.provider_id == "openai"


@pytest.mark.benchmark
def test_registry_capability_search(benchmark):
    """Benchmark capability-based search performance.

    This test measures the performance of searching for models by capabilities,
    such as tool calling support or reasoning support.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    registry = ModelRegistry.get_instance()

    # Benchmark searching for models with tool calling support
    def search_tool_models():
        return list(registry.search(supports_tool_call=True))

    results = benchmark(search_tool_models)

    # Verify results
    assert results is not None
    assert isinstance(results, list)
    # All results should support tool calling
    for model in results:
        assert model.capabilities.tool_call is True


@pytest.mark.benchmark
def test_chat_params_comparison(benchmark):
    """Benchmark ChatParams creation and comparison.

    This test measures the performance of creating ChatParams with different
    configurations and comparing their performance.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Benchmark creating params with explicit values
    def create_params_with_values():
        return ChatParams(
            temperature=0.5,
            max_tokens=500,
        )

    explicit_params = benchmark(create_params_with_values)
    assert explicit_params is not None
    assert explicit_params.temperature == 0.5
    assert explicit_params.max_tokens == 500


@pytest.mark.benchmark
def test_registry_cache_performance(benchmark):
    """Benchmark ModelRegistry cache performance.

    This test measures the performance benefit of the LRU cache for repeated
    model lookups. It compares cached vs uncached lookups to demonstrate
    the performance improvement.

    Args:
        benchmark: pytest-benchmark fixture for timing execution
    """
    # Get registry instance
    registry = ModelRegistry.get_instance()

    # Test model IDs for lookup
    test_models = ["gpt-4o", "gpt-4-turbo", "claude-3-opus", "gpt-3.5-turbo"]

    # First, populate the cache with initial lookups
    for model_id in test_models:
        registry.get(model_id, suppress_unknown_warning=True)

    # Benchmark cached lookups
    def cached_lookup():
        results = []
        for model_id in test_models:
            spec = registry.get(model_id, suppress_unknown_warning=True)
            results.append(spec)
        return results

    results = benchmark(cached_lookup)

    # Verify results
    assert results is not None
    assert len(results) == len(test_models)
    for spec in results:
        assert spec is not None
        assert hasattr(spec, "id")

    # Verify cache hits occurred
    cache_info = registry._get_model_spec_cached.cache_info()
    # Should have cache hits from the benchmark iterations
    assert cache_info.hits > 0
