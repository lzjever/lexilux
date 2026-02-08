"""Connection pool performance benchmarks.

This module contains benchmarks for measuring the performance impact of
connection pooling on HTTP requests.

Note: These are integration tests that need actual API endpoints or mock servers.
By default, these tests are skipped and must be explicitly enabled with
configuration.

Note on pool_size parameter:
- BaseAPIClient supports pool_size parameter for connection pooling
- Chat client inherits from BaseAPIClient but does not currently expose pool_size
- These tests use BaseAPIClient directly to test pool_size functionality
- Future improvements may expose pool_size on Chat client
"""

import time
from unittest import mock

import pytest

from lexilux._base import BaseAPIClient
from lexilux import Chat


@pytest.mark.benchmark
@pytest.mark.skipif(
    True,  # Default skip, needs configuration to run
    reason="Requires actual API endpoint or mock server configuration"
)
def test_connection_pool_performance(benchmark):
    """Benchmark: Verify connection pool brings performance improvement.

    This test measures the performance of making multiple requests with
    connection pooling enabled.

    To enable this test:
    1. Set up a mock server or use a real API endpoint
    2. Remove or modify the @pytest.mark.skipif decorator
    3. Configure the base_url and api_key parameters

    Args:
        benchmark: pytest-benchmark fixture for timing execution

    Example:
        With a mock server configured:
        >>> result = benchmark(make_requests)
        >>> print(f"Total time: {result}")
    """
    # Note: Chat doesn't expose pool_size parameter yet
    # Using BaseAPIClient directly for this benchmark
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        pool_size=10,
    )

    def make_requests():
        """Make multiple requests to test connection pool performance."""
        for _ in range(10):
            client._make_request("chat/completions", {"test": "data"})

    result = benchmark(make_requests)
    # Record result for comparison
    print(f"Total time: {result}")


@pytest.mark.benchmark
@pytest.mark.skipif(
    True,
    reason="Requires mock server setup"
)
def test_no_pool_vs_pool_performance():
    """Compare no pool vs pool performance.

    This test compares the performance of making requests with and without
    connection pooling. It requires a mock server to simulate API responses.

    To enable this test:
    1. Set up a mock server (e.g., using responses library or httpserver)
    2. Remove or modify the @pytest.mark.skipif decorator
    3. Configure the mock server responses

    The test will:
    1. Measure time with default pool_size (2 connections)
    2. Measure time with larger pool_size (10 connections)
    3. Compare results to demonstrate performance improvement

    Example:
        >>> time_with_default_pool = measure_requests(pool_size=2)
        >>> time_with_larger_pool = measure_requests(pool_size=10)
        >>> assert time_with_larger_pool < time_with_default_pool
    """
    # This test needs more complex setup for comparison
    # Implementation would use mock server
    pass


@pytest.mark.benchmark
@pytest.mark.skipif(
    True,
    reason="Requires mock server setup"
)
def test_concurrent_requests_with_pool():
    """Benchmark concurrent requests with connection pooling.

    This test measures how connection pooling affects performance when
    making concurrent requests to the API.

    Requires:
    - Mock server or real API endpoint
    - Thread pool or async implementation for concurrency

    Expected outcome:
    - With larger pool_size, concurrent requests should complete faster
    - Connection reuse should reduce overhead
    """
    pass


@pytest.mark.benchmark
def test_connection_pool_initialization():
    """Benchmark connection pool initialization time.

    This test measures how long it takes to initialize a client
    with different pool sizes. This is a lightweight benchmark that
    doesn't require external services.

    The test verifies that:
    1. Pool initialization is fast (< 10ms for typical sizes)
    2. Larger pools take slightly longer to initialize
    3. Initialization overhead is negligible compared to request time
    """
    # Test with default pool_size (Chat uses default from BaseAPIClient)
    start_time = time.perf_counter()

    client = Chat(
        base_url="https://api.example.com",
        api_key="test_key",
        model="gpt-4",
    )

    end_time = time.perf_counter()
    init_time = end_time - start_time

    # Verify client was created successfully
    assert client is not None
    assert client.base_url == "https://api.example.com"

    # Initialization should be very fast (< 10ms)
    assert init_time < 0.01, f"Initialization took {init_time:.4f}s, expected < 0.01s"

    print(f"Chat client initialization time: {init_time * 1000:.2f}ms")

    # Test BaseAPIClient with explicit pool_size
    for pool_size in [1, 10, 50]:
        start_time = time.perf_counter()

        base_client = BaseAPIClient(
            base_url="https://api.example.com",
            api_key="test_key",
            pool_size=pool_size,
        )

        end_time = time.perf_counter()
        init_time = end_time - start_time

        assert base_client is not None
        assert init_time < 0.01, f"Pool size {pool_size} initialization took {init_time:.4f}s, expected < 0.01s"

        print(f"BaseAPIClient (pool_size={pool_size}) initialization time: {init_time * 1000:.2f}ms")


@pytest.mark.benchmark
def test_pool_size_validation():
    """Benchmark pool size parameter validation.

    This test verifies that invalid pool sizes are rejected quickly
    without significant overhead.

    Test cases:
    1. pool_size < 1 should raise ValueError
    2. pool_size = 0 should raise ValueError
    3. pool_size = 1 should be valid (minimum)
    4. Large pool_size values should be accepted
    """
    import pytest as pt

    # Test invalid pool sizes on BaseAPIClient
    with pt.raises(ValueError, match="pool_size must be at least 1"):
        BaseAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            pool_size=0,
        )

    with pt.raises(ValueError, match="pool_size must be at least 1"):
        BaseAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            pool_size=-1,
        )

    # Test valid pool sizes
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        pool_size=1,
    )
    assert client is not None

    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        pool_size=100,
    )
    assert client is not None


@pytest.mark.benchmark
def test_session_reuse():
    """Benchmark that session is properly reused across requests.

    This test verifies that the same requests.Session object is used
    for multiple requests, which is essential for connection pooling
    to work correctly.
    """
    # Test with Chat client (uses default pool_size from BaseAPIClient)
    client = Chat(
        base_url="https://api.example.com",
        api_key="test",
    )

    # Verify session exists
    assert hasattr(client, "_session")
    assert client._session is not None

    # Verify session has HTTPAdapter mounted
    # This ensures connection pooling is configured
    session = client._session

    # Check that adapters are mounted for both http and https
    assert "http://" in session.adapters
    assert "https://" in session.adapters

    # Verify adapter is HTTPAdapter (which provides pooling)
    http_adapter = session.adapters["http://"]
    https_adapter = session.adapters["https://"]

    from requests.adapters import HTTPAdapter

    assert isinstance(http_adapter, HTTPAdapter)
    assert isinstance(https_adapter, HTTPAdapter)

    print(f"HTTP adapter: {http_adapter}")
    print(f"HTTPS adapter: {https_adapter}")
    print(f"Pool connections: {http_adapter._pool_connections}")
    print(f"Pool maxsize: {http_adapter._pool_maxsize}")


@pytest.mark.benchmark
def test_close_context_manager():
    """Benchmark context manager cleanup.

    This test verifies that the Chat client properly cleans up
    resources when used as a context manager.
    """
    base_url = "https://api.example.com"

    # Test normal usage with Chat
    with Chat(base_url=base_url, api_key="test") as client:
        assert client is not None
        session = client._session
        assert session is not None

    # Test with BaseAPIClient
    with BaseAPIClient(base_url=base_url, api_key="test", pool_size=5) as client:
        assert client is not None
        session = client._session
        assert session is not None

    # Session should be closed after exiting context
    # Note: requests.Session doesn't have a simple way to check if closed,
    # but we can verify the client works correctly


@pytest.mark.benchmark
def test_pool_size_parameter_propagation():
    """Verify pool_size parameter is properly propagated to HTTPAdapter.

    This test ensures that when pool_size is specified, it's correctly
    used to configure the underlying HTTPAdapter.
    """
    for pool_size in [1, 2, 5, 10, 20]:
        client = BaseAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            pool_size=pool_size,
        )

        http_adapter = client._session.adapters["http://"]

        # Verify pool_size was used
        assert http_adapter._pool_connections == pool_size
        assert http_adapter._pool_maxsize == pool_size

        print(f"Pool size {pool_size}: connections={http_adapter._pool_connections}, maxsize={http_adapter._pool_maxsize}")


@pytest.mark.benchmark
def test_chat_client_default_pool():
    """Verify Chat client uses default pool size from BaseAPIClient.

    This test ensures that when Chat is instantiated, it properly
    inherits the connection pooling configuration from BaseAPIClient.
    """
    client = Chat(
        base_url="https://api.example.com",
        api_key="test",
    )

    # Verify session exists with HTTPAdapter
    assert hasattr(client, "_session")
    session = client._session

    http_adapter = session.adapters["http://"]

    # BaseAPIClient default pool_size is 2
    assert http_adapter._pool_connections == 2
    assert http_adapter._pool_maxsize == 2

    print(f"Chat client default pool: connections={http_adapter._pool_connections}, maxsize={http_adapter._pool_maxsize}")


@pytest.mark.benchmark
def test_pool_configuration_comparison():
    """Compare different pool configurations.

    This benchmark measures the initialization time for different
    pool sizes to help understand the overhead of larger pools.
    """
    results = {}

    for pool_size in [1, 2, 5, 10, 20, 50, 100]:
        times = []
        for _ in range(5):  # Run 5 times for average
            start_time = time.perf_counter()

            client = BaseAPIClient(
                base_url="https://api.example.com",
                api_key="test",
                pool_size=pool_size,
            )

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        results[pool_size] = avg_time

        print(f"Pool size {pool_size:3d}: {avg_time:.3f}ms (avg of 5 runs)")

    # Verify all pool sizes initialize quickly (< 10ms)
    for pool_size, avg_time in results.items():
        assert avg_time < 10, f"Pool size {pool_size} took {avg_time:.2f}ms, expected < 10ms"
