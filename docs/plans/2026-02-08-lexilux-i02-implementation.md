# Lexilux I02@A02 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all P0-P3 issues from A02 code review through stability improvements, performance optimizations, and code quality enhancements in a single major iteration.

**Architecture:**
- Package 1 (Stability): Fix connection leaks, improve test coverage to 85%+, add concurrent tests, fix httpx connection params
- Package 2 (Performance): Add pytest-benchmark baselines, implement client-side rate limiting with aiolimiter, add LRU cache to ModelRegistry
- Package 3 (Code Quality): Refactor chat/client.py by functional responsibility, add lightweight input validation, convert comments to English, add SSL verification options

**Tech Stack:** Python 3.9+, pytest, pytest-asyncio, pytest-benchmark, aiolimiter, functools.lru_cache

**Principles:** KISS, DRY, SOLID, YAGNI, TDD, frequent commits

---

## Package 1: Stability (5.5 days)

### Task 1: Fix httpx Connection Parameters (P0-3)

**Files:**
- Modify: `lexilux/_base.py:530-540`

**Step 1: Read current httpx limits configuration**

Run: `grep -n "Limits" lexilux/_base.py`
Expected: Find `max_keepalive_connections=0`

**Step 2: Fix the connection limits**

In `lexilux/_base.py`, find the httpx Limits configuration around line 535:

Change from:
```python
limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
```

To:
```python
limits=httpx.Limits(
    max_connections=10,
    max_keepalive_connections=5,
),
```

**Step 3: Verify syntax**

Run: `python -m py_compile lexilux/_base.py`
Expected: No syntax errors

**Step 4: Run existing tests**

Run: `pytest tests/test_base.py -v`
Expected: All existing tests pass

**Step 5: Commit**

```bash
git add lexilux/_base.py
git commit -m "fix(I02): fix httpx connection limits to enable keepalive

Change max_connections from 1 to 10
Change max_keepalive_connections from 0 to 5

This enables HTTP connection reuse for async requests."
```

---

### Task 2: Add Connection Leak Context Manager (P0-2)

**Files:**
- Modify: `lexilux/_base.py`
- Modify: `lexilux/chat/client.py`
- Test: `tests/test_base.py`

**Step 1: Write failing test for context manager**

In `tests/test_base.py`, add:

```python
from contextlib import contextmanager

def test_streaming_request_context_closes_response():
    """Verify that streaming request context closes response."""
    client = BaseAPIClient(base_url="https://api.example.com", api_key="test")

    # Mock the session.post to return a response with close tracking
    class MockResponse:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock_response = MockResponse()

    with unittest.mock.patch.object(client._session, "post", return_value=mock_response):
        with client._streaming_request_context("test", {}):
            pass

    assert mock_response.closed, "Response should be closed after context exit"

def test_streaming_request_context_closes_on_exception():
    """Verify that streaming request context closes response even on exception."""
    client = BaseAPIClient(base_url="https://api.example.com", api_key="test")

    class MockResponse:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    mock_response = MockResponse()

    with unittest.mock.patch.object(client._session, "post", return_value=mock_response):
        with pytest.raises(ValueError):
            with client._streaming_request_context("test", {}):
                raise ValueError("Test exception")

    assert mock_response.closed, "Response should be closed even on exception"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_base.py::test_streaming_request_context_closes_response -v`
Expected: FAIL with "no attribute '_streaming_request_context'"

**Step 3: Implement context manager in BaseAPIClient**

In `lexilux/_base.py`, add imports at top:
```python
from contextlib import contextmanager
```

In `BaseAPIClient` class, add method after `_make_request`:

```python
@contextmanager
def _streaming_request_context(self, endpoint: str, payload: dict):
    """
    Context manager for streaming requests with guaranteed cleanup.

    Args:
        endpoint: API endpoint path
        payload: Request payload

    Yields:
        requests.Response: The response object

    Ensures response is closed even if exception occurs.
    """
    url = f"{self.base_url}/{endpoint}"
    response = self._session.post(
        url,
        json=payload,
        stream=True,
        timeout=self.timeout,
        headers=self.headers,
        proxies=self.proxies,
    )
    try:
        yield response
    finally:
        response.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_base.py::test_streaming_request_context_closes_response -v`
Run: `pytest tests/test_base.py::test_streaming_request_context_closes_on_exception -v`
Expected: PASS

**Step 5: Update Chat.stream() to use context manager**

In `lexilux/chat/client.py`, find the `stream()` method.

Look for the existing code that makes streaming request. Replace it with context manager usage.

The current implementation likely has:
```python
response = self._make_streaming_request("chat/completions", payload)
# ... iteration code
response.close()
```

Change to use the context manager (exact refactoring depends on current implementation).

**Step 6: Run chat tests to verify streaming still works**

Run: `pytest tests/chat/test_client.py -k stream -v`
Expected: All streaming tests pass

**Step 7: Commit**

```bash
git add lexilux/_base.py lexilux/chat/client.py tests/test_base.py
git commit -m "feat(I02): add streaming request context manager

Prevent connection leaks by using context manager that guarantees
response.close() is called even on exceptions or early iteration exit."
```

---

### Task 3: Add Concurrent Testing (P1-1)

**Files:**
- Create: `tests/test_concurrent.py`

**Step 1: Create concurrent test file**

Create `tests/test_concurrent.py`:

```python
"""Concurrent safety tests for lexilux clients."""
import asyncio
import threading
import pytest
from unittest.mock import Mock, patch
from lexilux import Chat


class TestConcurrentSync:
    """Test thread safety of sync clients."""

    def test_concurrent_sync_requests_same_client():
        """Test multiple threads using same Chat instance."""
        # Mock to avoid actual API calls
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"total_tokens": 10},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            results = []
            errors = []

            def make_request(n):
                try:
                    result = chat(f"Hello {n}")
                    results.append(result.content)
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(20):
                t = threading.Thread(target=make_request, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 20


class TestConcurrentAsync:
    """Test asyncio concurrency of async clients."""

    @pytest.mark.asyncio
    async def test_concurrent_async_requests_same_client():
        """Test multiple concurrent async requests."""
        with patch("lexilux._base.BaseAPIClient._ado_request") as mock_req:
            async def mock_async_request(*args, **kwargs):
                # Simulate async delay
                await asyncio.sleep(0.01)
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "id": "test",
                    "choices": [{"message": {"content": "Hello"}}],
                    "usage": {"total_tokens": 10},
                }
                return mock_response

            mock_req.side_effect = mock_async_request

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            async def make_request(n):
                result = await chat.astream(f"Hello {n}", max_tokens=10)
                # Collect first chunk
                async for _ in result:
                    break

            await asyncio.gather(*[make_request(i) for i in range(20)])

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests():
        """Test multiple concurrent streaming requests."""
        with patch("lexilux._base.BaseAPIClient._amake_streaming_request") as mock_stream:
            async def mock_stream_response(*args, **kwargs):
                # Simulate SSE stream
                async def stream_gen():
                    chunks = [
                        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
                        b'data: [DONE]\n\n',
                    ]
                    for chunk in chunks:
                        await asyncio.sleep(0.01)
                        yield chunk

                return stream_gen()

            mock_stream.return_value = mock_stream_response()

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            async def stream_request(n):
                count = 0
                async for _ in chat.astream(f"Hello {n}", max_tokens=10):
                    count += 1
                    if count >= 2:
                        break
                return count

            results = await asyncio.gather(*[stream_request(i) for i in range(10)])
            assert all(r >= 1 for r in results)
```

**Step 2: Run test to see current state**

Run: `pytest tests/test_concurrent.py -v`
Expected: May pass or fail - this establishes baseline

**Step 3: Fix any issues found**

If tests fail, investigate and fix. Common issues:
- Session object not thread-safe → requests.Session should be fine
- Race conditions in connection pool → may need locking
- Async state corruption → ensure proper async/await

**Step 4: Ensure tests pass**

Run: `pytest tests/test_concurrent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_concurrent.py
git commit -m "test(I02): add concurrent safety tests

Test thread safety for sync clients and asyncio safety for async clients.
Verify connection pool behaves correctly under concurrent load."
```

---

### Task 4: Improve Test Coverage - Boundary Conditions (P0-1)

**Files:**
- Create: `tests/chat/test_boundary_conditions.py`

**Step 1: Create boundary conditions test file**

Create `tests/chat/test_boundary_conditions.py`:

```python
"""Boundary condition tests for chat functionality."""
import pytest
from lexilux import Chat
from lexilux.exceptions import ValidationError


class TestInputBoundaries:
    """Test edge cases and boundary conditions for inputs."""

    def test_empty_string_message():
        """Test empty string message is handled."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 5},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")
            result = chat("")

            assert result is not None

    def test_very_long_message():
        """Test very long message doesn't break."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 100},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")
            long_message = "hello " * 10000
            result = chat(long_message)

            assert result is not None

    def test_zero_max_tokens():
        """Test max_tokens=0 is handled correctly."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": ""}}],
                "usage": {"total_tokens": 0},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")
            result = chat("Hello", max_tokens=0)

            # Verify the request was made with max_tokens=0
            call_args = mock_req.call_args
            payload = call_args[0][1]
            assert payload["max_tokens"] == 0

    def test_temperature_at_boundaries():
        """Test temperature at valid boundaries."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 5},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            # Test minimum temperature
            chat("Hello", temperature=0.0)
            call_args = mock_req.call_args
            assert call_args[0][1]["temperature"] == 0.0

            # Test maximum temperature
            chat("Hello", temperature=2.0)
            call_args = mock_req.call_args
            assert call_args[0][1]["temperature"] == 2.0

    def test_invalid_temperature_raises_error():
        """Test invalid temperature raises validation error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with pytest.raises(ValidationError):
            chat("Hello", temperature=3.0)

        with pytest.raises(ValidationError):
            chat("Hello", temperature=-0.1)

    def test_many_messages_in_history():
        """Test large message history."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 100},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            # Create 100 message history
            messages = [
                {"role": "user", "content": f"Message {i}"}
                for i in range(100)
            ]
            result = chat(messages)

            assert result is not None
            # Verify all messages were sent
            call_args = mock_req.call_args
            payload = call_args[0][1]
            assert len(payload["messages"]) == 100

    def test_n_parameter_at_boundaries():
        """Test n parameter at valid boundaries."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            # Return multiple choices
            mock_response.json.return_value = {
                "id": "test",
                "choices": [
                    {"message": {"content": f"Choice {i}"}}
                    for i in range(5)
                ],
                "usage": {"total_tokens": 20},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            # Test n=1 (minimum)
            result = chat("Hello", n=1)
            assert len(result.choices) == 1

            # Test n=5
            result = chat("Hello", n=5)
            assert len(result.choices) == 5

    def test_invalid_n_raises_error():
        """Test invalid n raises validation error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with pytest.raises(ValidationError):
            chat("Hello", n=0)

        with pytest.raises(ValidationError):
            chat("Hello", n=-1)
```

**Step 2: Run tests to see current state**

Run: `pytest tests/chat/test_boundary_conditions.py -v`
Expected: Some may fail if validation not implemented yet

**Step 3: Note coverage increase**

Run: `pytest --cov=lexilux --cov-report=term-missing tests/chat/test_boundary_conditions.py`
Expected: Coverage increase from new tests

**Step 4: Commit**

```bash
git add tests/chat/test_boundary_conditions.py
git commit -m "test(I02): add boundary condition tests

Test empty inputs, very long inputs, zero max_tokens,
temperature boundaries, n parameter boundaries,
and large message histories."
```

---

### Task 5: Improve Test Coverage - Exception Paths (P0-1)

**Files:**
- Create: `tests/chat/test_exception_paths.py`

**Step 1: Create exception paths test file**

Create `tests/chat/test_exception_paths.py`:

```python
"""Exception path tests for error handling."""
import pytest
import requests
from lexilux import Chat
from lexilux.exceptions import (
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    ConnectionError,
    ServerError,
)


class TestNetworkExceptions:
    """Test network-related exception handling."""

    def test_timeout_exception():
        """Test timeout is properly raised."""
        chat = Chat(
            base_url="https://api.example.com",
            api_key="test",
            model="gpt-4",
            timeout_s=0.001,  # Very short timeout
        )

        with pytest.raises(TimeoutError):
            chat("Hello")

    def test_connection_refused():
        """Test connection refused is properly raised."""
        chat = Chat(
            base_url="https://localhost:9999",  # Non-existent server
            api_key="test",
            model="gpt-4",
            timeout_s=1,
        )

        with pytest.raises(ConnectionError):
            chat("Hello")


class TestAPIExceptions:
    """Test API error response handling."""

    def test_401_unauthorized():
        """Test 401 response raises AuthenticationError."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {
                "error": {"message": "Invalid API key"}
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(AuthenticationError):
                chat("Hello")

    def test_429_rate_limit():
        """Test 429 response raises RateLimitError."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": {"message": "Rate limit exceeded"}
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(RateLimitError):
                chat("Hello")

    def test_500_server_error():
        """Test 500 response raises ServerError."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {
                "error": {"message": "Internal server error"}
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(ServerError):
                chat("Hello")

    def test_503_service_unavailable():
        """Test 503 response raises ServerError."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_response.json.return_value = {
                "error": {"message": "Service unavailable"}
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(ServerError):
                chat("Hello")


class TestMalformedResponses:
    """Test handling of malformed API responses."""

    def test_empty_response_body():
        """Test empty response body is handled."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}  # Empty response
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(Exception):  # May raise various exceptions
                chat("Hello")

    def test_missing_choices_in_response():
        """Test response without choices is handled."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                # Missing "choices" key
                "usage": {"total_tokens": 10},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(Exception):
                chat("Hello")

    def test_invalid_json_response():
        """Test invalid JSON response is handled."""
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            with pytest.raises(Exception):
                chat("Hello")
```

**Step 2: Run tests to verify coverage**

Run: `pytest tests/chat/test_exception_paths.py -v`
Expected: Tests should pass with current exception handling

**Step 3: Check coverage**

Run: `pytest --cov=lexilux --cov-report=term-missing tests/chat/test_exception_paths.py`
Expected: Increased coverage

**Step 4: Commit**

```bash
git add tests/chat/test_exception_paths.py
git commit -m "test(I02): add exception path tests

Test network errors (timeout, connection refused),
API errors (401, 429, 500, 503),
and malformed responses."
```

---

### Task 6: Verify 85% Coverage Target (P0-1)

**Step 1: Run full coverage report**

Run: `pytest --cov=lexilux --cov-report=term-missing --cov-report=html`
Expected: See current coverage percentage

**Step 2: Identify uncovered lines**

Review the terminal output showing missing lines per file.

**Step 3: Add targeted tests for missing coverage**

For each file with low coverage, add specific tests.

Example for `lexilux/registry/registry.py`:

Create `tests/registry/test_registry_edge_cases.py`:

```python
"""Edge case tests for ModelRegistry."""
import pytest
from lexilux.registry import ModelRegistry
from lexilux.exceptions import UnknownModelError


class TestModelRegistryEdgeCases:
    """Test edge cases in model registry."""

    def test_unknown_model raises_error():
        """Test requesting unknown model raises error."""
        registry = ModelRegistry()
        with pytest.raises(UnknownModelError):
            registry.get_model_spec("definitely-not-a-real-model-xyz-123")

    def test_case_sensitive_model_lookup():
        """Test model lookup is case sensitive."""
        registry = ModelRegistry()
        # GPT-4 should work
        spec = registry.get_model_spec("gpt-4")
        assert spec is not None

        # gpt-4 (different case) should fail or return different result
        with pytest.raises(UnknownModelError):
            registry.get_model_spec("GPT-4")

    def test_provider_filtering():
        """Test filtering by provider."""
        registry = ModelRegistry()

        # Get all OpenAI models
        openai_models = registry.list_models(provider="openai")
        assert all(m.provider == "openai" for m in openai_models)

    def test_get_all_limits():
        """Test getting limits for all models."""
        registry = ModelRegistry()
        limits = registry.get_all_limits()
        assert isinstance(limits, dict)
        assert len(limits) > 0
```

**Step 4: Re-run coverage**

Run: `pytest --cov=lexilux --cov-report=term-missing`
Expected: Coverage >= 85%

**Step 5: If still below 85%, add more tests**

Repeat steps 2-4 until target is reached.

**Step 6: Commit**

```bash
git add tests/
git commit -m "test(I02): additional tests to reach 85% coverage target

Add edge case tests for ModelRegistry and other modules
to achieve 85%+ test coverage."
```

---

## Package 2: Performance (4 days)

### Task 7: Setup Performance Benchmarking (P1-2)

**Files:**
- Modify: `pyproject.toml`
- Create: `benchmarks/test_chat_performance.py`
- Create: `benchmarks/conftest.py`

**Step 1: Add pytest-benchmark dependency**

Edit `pyproject.toml`, add to optional dependencies:

```toml
[project.optional-dependencies]
benchmark = [
    "pytest-benchmark>=4.0.0",
]
```

**Step 2: Create benchmarks conftest**

Create `benchmarks/conftest.py`:

```python
"""Configuration for benchmark tests."""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
```

**Step 3: Create chat performance benchmarks**

Create `benchmarks/test_chat_performance.py`:

```python
"""Performance benchmarks for Chat API."""
import pytest
from lexilux import Chat


class TestChatPerformance:
    """Performance benchmarks for chat operations."""

    @pytest.mark.benchmark
    def test_chat_latency_single_request(self, benchmark):
        """Benchmark latency of single chat request."""
        # Use mock to avoid actual API calls in benchmarks
        from unittest.mock import Mock, patch

        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "Hello, world!"}}],
                "usage": {"total_tokens": 10},
            }
            mock_req.return_value = mock_response

            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            result = benchmark(chat, "Hello, world!")
            assert result.content == "Hello, world!"

    @pytest.mark.benchmark
    def test_chat_payload_building(self, benchmark):
        """Benchmark payload building performance."""
        from unittest.mock import patch

        with patch("lexilux._base.BaseAPIClient._make_request"):
            chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

            def build_payload():
                chat._build_payload(
                    "Test message",
                    temperature=0.7,
                    max_tokens=100,
                )

            benchmark(build_payload)

    @pytest.mark.benchmark
    def test_model_registry_lookup(self, benchmark):
        """Benchmark ModelRegistry lookup performance."""
        from lexilux.registry import ModelRegistry

        registry = ModelRegistry()

        def lookup_model():
            registry.get_model_spec("gpt-4")

        benchmark(lookup_model)
```

**Step 4: Install benchmark dependencies**

Run: `uv sync --group benchmark`
Expected: pytest-benchmark installed

**Step 5: Run benchmarks**

Run: `pytest benchmarks/ -v --benchmark-only`
Expected: Benchmarks execute successfully

**Step 6: Commit**

```bash
git add pyproject.toml benchmarks/conftest.py benchmarks/test_chat_performance.py
git commit -m "feat(I02): add performance benchmarking infrastructure

Add pytest-benchmark for latency and throughput measurements.
Establish baseline metrics for chat operations."
```

---

### Task 8: Implement Client-Side Rate Limiting (P2-1)

**Files:**
- Modify: `pyproject.toml`
- Create: `lexilux/_rate_limit.py`
- Modify: `lexilux/chat/client.py`
- Test: `tests/test_rate_limit.py`

**Step 1: Add aiolimiter dependency**

Edit `pyproject.toml`:

```toml
[project.optional-dependencies]
rate-limit = [
    "aiolimiter>=1.1.0",
]
```

**Step 2: Create rate limiter module**

Create `lexilux/_rate_limit.py`:

```python
"""Client-side rate limiting for API requests."""
from aiolimiter import AsyncLimiter


class RateLimiter:
    """
    Client-side rate limiter to prevent API quota exhaustion.

    Uses token bucket algorithm via aiolimiter.
    """

    def __init__(self, rate_limit: int = 60, time_period: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum number of requests allowed
            time_period: Time window in seconds
        """
        self._limiter = AsyncLimiter(rate_limit, time_period)
        self._rate_limit = rate_limit
        self._time_period = time_period

    async def acquire(self):
        """Acquire permission to make a request. Blocks if limit reached."""
        await self._limiter.acquire()

    @property
    def rate_limit(self) -> int:
        """Get the configured rate limit."""
        return self._rate_limit

    @property
    def time_period(self) -> float:
        """Get the configured time period."""
        return self._time_period
```

**Step 3: Write tests for rate limiter**

Create `tests/test_rate_limit.py`:

```python
"""Test rate limiting functionality."""
import pytest
import asyncio
import time
from lexilux._rate_limit import RateLimiter


class TestRateLimiter:
    """Test RateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_respects_limit(self):
        """Test that rate limiter enforces the rate limit."""
        limiter = RateLimiter(rate_limit=5, time_period=1.0)

        start = time.time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.time() - start

        # First 5 should be fast
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_limit_reached(self):
        """Test that rate limiter blocks after limit is reached."""
        limiter = RateLimiter(rate_limit=2, time_period=1.0)

        start = time.time()
        for _ in range(4):
            await limiter.acquire()
        elapsed = time.time() - start

        # Should take at least 1 second for 4 requests (2 per second)
        assert elapsed >= 0.9  # Small buffer for timing

    @pytest.mark.asyncio
    async def test_rate_limiter_properties(self):
        """Test rate limiter property accessors."""
        limiter = RateLimiter(rate_limit=100, time_period=60.0)

        assert limiter.rate_limit == 100
        assert limiter.time_period == 60.0
```

**Step 4: Run tests to verify rate limiter works**

Run: `pytest tests/test_rate_limit.py -v`
Expected: PASS

**Step 5: Integrate rate limiter into Chat client**

In `lexilux/chat/client.py`, add import:

```python
from lexilux._rate_limit import RateLimiter
```

In `Chat.__init__`, add rate_limit parameter:

```python
def __init__(
    self,
    *,
    base_url: str,
    api_key: str | None = None,
    model: str,
    rate_limit: int | None = None,
    # ... existing parameters
):
    """
    Initialize Chat client.

    Args:
        base_url: Base URL for API
        api_key: API key for authentication
        model: Default model to use
        rate_limit: Optional rate limit (requests per second)
        ... other args
    """
    # ... existing init code
    self._rate_limiter = RateLimiter(rate_limit or 60, 1.0) if rate_limit else None
```

Update `_amake_request` to use rate limiter (if async):

```python
async def _amake_request(self, endpoint: str, payload: dict):
    """Make async request with rate limiting."""
    if self._rate_limiter:
        await self._rate_limiter.acquire()
    return await super()._amake_request(endpoint, payload)
```

**Step 6: Test integration with Chat**

Create test in `tests/chat/test_rate_limit_integration.py`:

```python
"""Test rate limiting integration with Chat."""
import pytest
from unittest.mock import Mock, patch
from lexilux import Chat


class TestChatRateLimiting:
    """Test Chat client rate limiting."""

    @pytest.mark.asyncio
    async def test_chat_respects_rate_limit(self):
        """Test that Chat client respects configured rate limit."""
        with patch("lexilux._base.BaseAPIClient._ado_request") as mock_req:
            async def mock_async_request(*args, **kwargs):
                await asyncio.sleep(0.01)
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "id": "test",
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"total_tokens": 5},
                }
                return mock_response

            mock_req.side_effect = mock_async_request

            chat = Chat(
                base_url="https://api.example.com",
                api_key="test",
                model="gpt-4",
                rate_limit=5,  # 5 requests per second
            )

            import time
            start = time.time()
            for _ in range(10):
                await chat.astream("test", max_tokens=1)
                # Consume one chunk
                async for _ in chat.astream("test", max_tokens=1):
                    break
            elapsed = time.time() - start

            # Should take at least 1 second for 10 requests at 5/sec
            assert elapsed >= 1.0
```

**Step 7: Run all tests**

Run: `pytest tests/test_rate_limit.py tests/chat/test_rate_limit_integration.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add lexilux/_rate_limit.py lexilux/chat/client.py pyproject.toml tests/test_rate_limit.py tests/chat/test_rate_limit_integration.py
git commit -m "feat(I02): add client-side rate limiting

Add optional rate limiting using aiolimiter to prevent
API quota exhaustion. Configured via rate_limit parameter."
```

---

### Task 9: Add ModelRegistry LRU Cache (P2-3)

**Files:**
- Modify: `lexilux/registry/registry.py`
- Test: `tests/registry/test_registry_cache.py`

**Step 1: Write failing test for caching**

Create `tests/registry/test_registry_cache.py`:

```python
"""Test ModelRegistry caching."""
import pytest
from lexilux.registry import ModelRegistry


class TestModelRegistryCache:
    """Test ModelRegistry LRU cache functionality."""

    def test_cached_model_lookup_is_faster(self):
        """Test that cached lookups are faster."""
        registry = ModelRegistry()

        # First lookup
        spec1 = registry.get_model_spec("gpt-4")

        # Second lookup (should be cached)
        spec2 = registry.get_model_spec("gpt-4")

        assert spec1 is spec2 or spec1.model_id == spec2.model_id

    def test_cache_hits_for_repeated_lookups(self):
        """Test that repeated model lookups hit cache."""
        registry = ModelRegistry()

        # Look up same model multiple times
        for _ in range(10):
            spec = registry.get_model_spec("gpt-4")
            assert spec.model_id == "gpt-4"

    def test_unknown_model_not_cached(self):
        """Test that unknown models are not cached."""
        from lexilux.exceptions import UnknownModelError

        registry = ModelRegistry()

        with pytest.raises(UnknownModelError):
            registry.get_model_spec("definitely-not-real-model-xyz")

        # Should still raise on second try (not cached as error)
        with pytest.raises(UnknownModelError):
            registry.get_model_spec("definitely-not-real-model-xyz")
```

**Step 2: Run test to see baseline**

Run: `pytest tests/registry/test_registry_cache.py -v`
Expected: May pass (current implementation is consistent)

**Step 3: Add LRU cache to ModelRegistry**

In `lexilux/registry/registry.py`, add import:

```python
from functools import lru_cache
```

Add cached method:

```python
class ModelRegistry:
    # ... existing code

    @lru_cache(maxsize=128)
    def _get_model_spec_cached(self, model_id: str) -> ModelSpec:
        """
        Cached version of model spec lookup.

        Args:
            model_id: Model identifier

        Returns:
            ModelSpec for the model

        Raises:
            UnknownModelError: If model not found
        """
        return self._get_model_spec(model_id)
```

Update `get_model_spec` to use cached version:

```python
def get_model_spec(self, model_id: str) -> ModelSpec:
    """
    Get model specification by ID.

    Args:
        model_id: Model identifier (e.g., "gpt-4", "claude-3-opus")

    Returns:
        ModelSpec for the requested model

    Raises:
        UnknownModelError: If model not found in registry
    """
    normalized_id = model_id.lower().strip()

    try:
        return self._get_model_spec_cached(normalized_id)
    except UnknownModelError:
        # Re-raise with warning
        self.logger.warning(
            "Unknown model '%s'. Known models: %s",
            model_id,
            list(self._models.keys())[:10],  # Show first 10
        )
        raise
```

**Step 4: Run tests to verify caching works**

Run: `pytest tests/registry/test_registry_cache.py -v`
Expected: PASS

**Step 5: Verify cache performance**

Add benchmark test:

```python
# In benchmarks/test_chat_performance.py

@pytest.mark.benchmark
def test_model_registry_cached_lookup(benchmark):
    """Benchmark cached ModelRegistry lookup."""
    from lexilux.registry import ModelRegistry

    registry = ModelRegistry()

    # Prime the cache
    registry.get_model_spec("gpt-4")

    # Benchmark cached lookup
    benchmark(registry.get_model_spec, "gpt-4")
```

Run: `pytest benchmarks/ -k cached_lookup -v --benchmark-only`
Expected: Cached lookups are very fast

**Step 6: Commit**

```bash
git add lexilux/registry/registry.py tests/registry/test_registry_cache.py benchmarks/test_chat_performance.py
git commit -m "perf(I02): add LRU cache to ModelRegistry

Cache model specifications using functools.lru_cache.
Reduces redundant file I/O for repeated model lookups.
Cache size: 128 models."
```

---

## Package 3: Code Quality (4 days)

### Task 10: Refactor chat/client.py - Extract streaming.py (P1-4)

**Files:**
- Create: `lexilux/chat/streaming.py`
- Modify: `lexilux/chat/client.py`

**Step 1: Identify streaming-related code**

In `lexilux/chat/client.py`, identify:
- `stream()` method
- `astream()` method
- `StreamingIterator` class
- Any helper functions for streaming

**Step 2: Create new streaming.py module**

Create `lexilux/chat/streaming.py`:

```python
"""Streaming functionality for Chat API."""
from typing import Iterator, AsyncIterator
from lexilux.chat.models import ChatStreamChunk
from lexilux.chat.streaming_parser import SSEChatStreamParser


class StreamingIterator:
    """
    Iterator for streaming chat responses.

    Handles SSE (Server-Sent Events) parsing and chunk yielding.
    """

    def __init__(self, generator: Iterator[ChatStreamChunk]):
        """
        Initialize streaming iterator.

        Args:
            generator: Generator yielding chat stream chunks
        """
        self._generator = generator

    def __iter__(self) -> Iterator[ChatStreamChunk]:
        """Return iterator."""
        return self._generator

    def __next__(self) -> ChatStreamChunk:
        """Get next chunk."""
        return next(self._generator)

    def to_list(self) -> list[ChatStreamChunk]:
        """
        Collect all chunks into a list.

        Returns:
            List of all chunks from the stream
        """
        return list(self._generator)


class AsyncStreamingIterator:
    """
    Async iterator for streaming chat responses.
    """

    def __init__(self, generator: AsyncIterator[ChatStreamChunk]):
        """
        Initialize async streaming iterator.

        Args:
            generator: Async generator yielding chat stream chunks
        """
        self._generator = generator

    def __aiter__(self) -> AsyncIterator[ChatStreamChunk]:
        """Return async iterator."""
        return self._generator

    async def __anext__(self) -> ChatStreamChunk:
        """Get next chunk."""
        return await self._generator.__anext__()

    async def to_list(self) -> list[ChatStreamChunk]:
        """
        Collect all chunks into a list.

        Returns:
            List of all chunks from the stream
        """
        chunks = []
        async for chunk in self._generator:
            chunks.append(chunk)
        return chunks
```

**Step 3: Update client.py to use streaming module**

In `lexilux/chat/client.py`, add import:

```python
from lexilux.chat.streaming import StreamingIterator, AsyncStreamingIterator
```

Update `stream()` method to use new class:

```python
def stream(self, messages, **params) -> StreamingIterator:
    """
    Stream chat completion response.

    Args:
        messages: User messages
        **params: Additional chat parameters

    Returns:
        StreamingIterator yielding response chunks
    """
    payload = self._build_payload(messages, stream=True, **params)

    def _chunk_generator() -> Iterator[ChatStreamChunk]:
        """Internal generator for streaming chunks."""
        parser = SSEChatStreamParser(
            return_raw_events=params.get("return_raw_events", False),
            include_reasoning=params.get("include_reasoning", False),
        )

        with self._streaming_request_context("chat/completions", payload) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    line_str = line.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                chunk = parser.feed_line(line_str)
                if chunk is None:
                    continue
                yield chunk
                if parser.done:
                    break

    return StreamingIterator(_chunk_generator())
```

**Step 4: Update astream() similarly**

Update `astream()` to use `AsyncStreamingIterator`.

**Step 5: Run tests to verify refactoring didn't break anything**

Run: `pytest tests/chat/test_client.py -k stream -v`
Expected: All streaming tests pass

**Step 6: Commit**

```bash
git add lexilux/chat/streaming.py lexilux/chat/client.py
git commit -m "refactor(I02): extract streaming functionality to separate module

Move StreamingIterator and streaming logic from client.py
to dedicated streaming.py module for better separation of concerns."
```

---

### Task 11: Refactor chat/client.py - Extract continue.py (P1-4)

**Files:**
- Create: `lexilux/chat/continue.py`
- Modify: `lexilux/chat/client.py`

**Step 1: Create continue.py module**

Create `lexilux/chat/continue.py`:

```python
"""Conversation continuation functionality."""
from typing import TYPE_CHECKING
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult

if TYPE_CHECKING:
    from lexilux.chat.client import Chat


class ConversationContinuer:
    """
    Handles conversation continuation logic.

    Manages working history and state for complete() methods.
    """

    def __init__(self, client: "Chat"):
        """
        Initialize conversation continuer.

        Args:
            client: Parent Chat client instance
        """
        self._client = client

    def complete(
        self,
        history: ChatHistory,
        **params
    ) -> tuple[ChatResult, ChatHistory]:
        """
        Complete conversation with assistant message.

        Args:
            history: Conversation history
            **params: Additional chat parameters

        Returns:
            Tuple of (result, updated_history)
        """
        # Create working copy to avoid modifying original
        working_history = history.copy()

        # Make request with current history
        result = self._client(working_history.to_messages(), **params)

        # Add assistant response to history
        working_history.add_assistant_message(result.content)

        return result, working_history

    def complete_stream(
        self,
        history: ChatHistory,
        **params
    ) -> tuple:
        """
        Complete conversation with streaming response.

        Args:
            history: Conversation history
            **params: Additional chat parameters

        Returns:
            Tuple of (streaming_iterator, updated_history_future)
        """
        from lexilux.chat.streaming import AsyncStreamingIterator

        # Create working copy
        working_history = history.copy()

        # Start streaming request
        stream = self._client.stream(working_history.to_messages(), stream=True, **params)

        # Create async future for history update
        async def update_history():
            content = ""
            async for chunk in stream:
                if chunk.content:
                    content += chunk.content
            working_history.add_assistant_message(content)
            return working_history

        return stream, update_history()
```

**Step 2: Update client.py to use continue module**

In `lexilux/chat/client.py`, add:

```python
from lexilux.chat.continue import ConversationContinuer
```

In `Chat.__init__`:

```python
self._continuer = ConversationContinuer(self)
```

Update `complete()` method:

```python
def complete(self, history: ChatHistory, **params) -> tuple[ChatResult, ChatHistory]:
    """
    Complete conversation and return result with updated history.

    Args:
        history: Conversation history
        **params: Additional chat parameters

    Returns:
        Tuple of (chat_result, updated_history)
    """
    return self._continuer.complete(history, **params)
```

**Step 3: Run tests**

Run: `pytest tests/chat/test_continue.py -v`
Expected: All continuation tests pass

**Step 4: Commit**

```bash
git add lexilux/chat/continue.py lexilux/chat/client.py
git commit -m "refactor(I02): extract continuation logic to separate module

Move conversation continuation functionality from client.py
to dedicated continue.py module."
```

---

### Task 12: Refactor chat/client.py - Extract validation.py (P1-4)

**Files:**
- Create: `lexilux/chat/validation.py`
- Modify: `lexilux/chat/client.py`
- Modify: `lexilux/exceptions.py`

**Step 1: Add ValidationError to exceptions**

In `lexilux/exceptions.py`, add:

```python
class ValidationError(LexiluxError):
    """
    Raised when input validation fails.

    Attributes:
        message: Human-readable error message
        code: Error code ("validation_error")
        retryable: False (validation errors should not retry)
    """
    code = "validation_error"
    retryable = False
```

**Step 2: Create validation.py module**

Create `lexilux/chat/validation.py`:

```python
"""Input validation for Chat API."""
from lexilux.chat.params import ChatParams
from lexilux.exceptions import ValidationError


def validate_chat_params(params: ChatParams) -> None:
    """
    Validate chat parameters.

    Args:
        params: ChatParams to validate

    Raises:
        ValidationError: If validation fails
    """
    # Temperature validation
    if params.temperature is not None:
        if not 0.0 <= params.temperature <= 2.0:
            raise ValidationError(
                f"temperature must be between 0 and 2, got {params.temperature}"
            )

    # Max tokens validation
    if params.max_tokens is not None:
        if params.max_tokens < 0:
            raise ValidationError(
                f"max_tokens must be non-negative, got {params.max_tokens}"
            )

    # N validation
    if params.n is not None and params.n < 1:
        raise ValidationError(
            f"n must be at least 1, got {params.n}"
        )

    # Top_p validation
    if params.top_p is not None:
        if not 0.0 <= params.top_p <= 1.0:
            raise ValidationError(
                f"top_p must be between 0 and 1, got {params.top_p}"
            )

    # Presence_penalty validation
    if params.presence_penalty is not None:
        if not -2.0 <= params.presence_penalty <= 2.0:
            raise ValidationError(
                f"presence_penalty must be between -2 and 2, got {params.presence_penalty}"
            )

    # Frequency_penalty validation
    if params.frequency_penalty is not None:
        if not -2.0 <= params.frequency_penalty <= 2.0:
            raise ValidationError(
                f"frequency_penalty must be between -2 and 2, got {params.frequency_penalty}"
            )


def validate_messages(messages: list | str) -> list:
    """
    Validate and normalize messages input.

    Args:
        messages: String message or list of message dicts

    Returns:
        Normalized list of message dicts

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if not messages:
        raise ValidationError("messages cannot be empty")

    if not isinstance(messages, list):
        raise ValidationError(
            f"messages must be list or str, got {type(messages).__name__}"
        )

    # Validate each message
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(
                f"message at index {i} must be dict, got {type(msg).__name__}"
            )
        if "role" not in msg:
            raise ValidationError(
                f"message at index {i} missing 'role' field"
            )
        if "content" not in msg:
            raise ValidationError(
                f"message at index {i} missing 'content' field"
            )

    return messages


def validate_model(model: str) -> None:
    """
    Validate model identifier.

    Args:
        model: Model identifier

    Raises:
        ValidationError: If validation fails
    """
    if not model or not isinstance(model, str):
        raise ValidationError(
            f"model must be a non-empty string, got {model!r}"
        )

    if not model.strip():
        raise ValidationError("model cannot be empty or whitespace")
```

**Step 3: Update client.py to use validation**

In `lexilux/chat/client.py`, add imports:

```python
from lexilux.chat.validation import (
    validate_chat_params,
    validate_messages,
    validate_model,
)
```

Update `__call__` method:

```python
def __call__(self, messages, **params):
    """
    Make a chat completion request.

    Args:
        messages: User messages (string or list of message dicts)
        **params: Additional chat parameters

    Returns:
        ChatResult with the response

    Raises:
        ValidationError: If input validation fails
    """
    # Validate inputs
    validate_model(self.model)
    normalized_messages = validate_messages(messages)
    chat_params = ChatParams(**params)
    validate_chat_params(chat_params)

    # Build and make request
    payload = self._build_payload(normalized_messages, **params)
    response = self._make_request("chat/completions", payload)
    return parse_chat_completion_response(response, chat_params)
```

**Step 4: Run tests**

Run: `pytest tests/chat/test_client.py -v`
Expected: Tests pass, including new validation tests

**Step 5: Commit**

```bash
git add lexilux/chat/validation.py lexilux/chat/client.py lexilux/exceptions.py
git commit -m "refactor(I02): extract validation logic to separate module

Move input validation from client.py to dedicated validation.py module.
Add comprehensive parameter validation with clear error messages."
```

---

### Task 13: Convert Comments to English (P3-1)

**Files:**
- Modify: All files with non-English comments

**Step 1: Find all non-English comments**

Run: `grep -rn "[\u4e00-\u9fff]" lexilux/ --include="*.py"`
Expected: List of files with Chinese/other non-ASCII comments

**Step 2: Convert known Chinese comment**

In `lexilux/chat/_request.py:279` (or similar), find any Chinese comments.

Example change:
```python
# Before (Chinese):
# 处理多模态内容

# After (English):
# Handle multimodal content
```

**Step 3: Review and translate all non-English comments**

For each file found, translate comments to clear, concise English while maintaining technical accuracy.

**Step 4: Verify no syntax errors**

Run: `python -m py_compile lexilux/**/*.py`
Expected: No syntax errors

**Step 5: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add lexilux/
git commit -m "style(I02): convert all comments to English

Standardize code comments to English for international collaboration.
Ensure technical accuracy is maintained in translations."
```

---

### Task 14: Add SSL Verification Options (P3-2)

**Files:**
- Modify: `lexilux/_base.py`
- Test: `tests/test_base.py`

**Step 1: Write failing test for SSL verification**

In `tests/test_base.py`, add:

```python
def test_ssl_verification_option():
    """Test SSL verification option is passed through."""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        verify_ssl="/path/to/ca-bundle.crt"
    )

    assert client._verify_ssl == "/path/to/ca-bundle.crt"

def test_ssl_verification_disabled():
    """Test SSL verification can be disabled."""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        verify_ssl=False
    )

    assert client._verify_ssl is False

def test_ssl_verification_default():
    """Test SSL verification defaults to True."""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test"
    )

    assert client._verify_ssl is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_base.py::test_ssl_verification_option -v`
Expected: FAIL (verify_ssl parameter doesn't exist)

**Step 3: Add verify_ssl parameter to BaseAPIClient**

In `lexilux/_base.py`, update `__init__` signature:

```python
def __init__(
    self,
    *,
    base_url: str,
    api_key: str | None = None,
    timeout_s: float = 60.0,
    connect_timeout_s: float | None = None,
    read_timeout_s: float | None = None,
    max_retries: int = 0,
    headers: dict[str, str] | None = None,
    proxies: dict[str, str] | None = None,
    pool_size: int = 2,
    verify_ssl: str | bool = True,  # NEW PARAMETER
):
    """
    Initialize base API client.

    Args:
        base_url: Base URL for API endpoints
        api_key: API key for authentication
        timeout_s: Default timeout for requests (seconds)
        connect_timeout_s: Connection timeout (seconds)
        read_timeout_s: Read timeout (seconds)
        max_retries: Maximum number of retries for failed requests
        headers: Default headers to include with requests
        proxies: Proxy configuration
        pool_size: Connection pool size
        verify_ssl: SSL verification (True=default, False=disable, or path to CA bundle)
    """
    self.base_url = base_url.rstrip("/")
    self._api_key = api_key
    # ... existing timeout setup ...
    self._max_retries = max_retries
    self._headers = headers or {}
    self._proxies = proxies

    # NEW: Store SSL verification setting
    self._verify_ssl = verify_ssl

    # ... existing session setup ...
```

**Step 4: Update _make_request to use verify_ssl**

In `_make_request` method:

```python
def _make_request(self, endpoint: str, payload: dict) -> requests.Response:
    """Make HTTP POST request with retry logic."""
    url = f"{self.base_url}/{endpoint}"

    retry_decorator = self._get_retry_decorator(self._max_retries + 1)
    request_func = retry_decorator(self._do_request)

    try:
        return request_func(endpoint, payload)
    except requests.exceptions.RequestException as e:
        raise self._map_exception(e)
```

Update `_do_request` to use verify_ssl:

```python
def _do_request(self, endpoint: str, payload: dict) -> requests.Response:
    """Execute the actual HTTP request."""
    url = f"{self.base_url}/{endpoint}"
    return self._session.post(
        url,
        json=payload,
        timeout=self.timeout,
        headers=self.headers,
        proxies=self.proxies,
        verify=self._verify_ssl,  # NEW: Use verify_ssl setting
    )
```

**Step 5: Update async method similarly**

Update `_ado_request` for async:

```python
async def _ado_request(self, endpoint: str, payload: dict):
    """Execute async request."""
    url = f"{self.base_url}/{endpoint}"
    async with httpx.AsyncClient(
        timeout=self.timeout,
        headers=self.headers,
        proxies=self.proxies,
        verify=self._verify_ssl,  # NEW: Use verify_ssl setting
    ) as client:
        response = await client.post(url, json=payload)
        return response
```

**Step 6: Run tests to verify implementation**

Run: `pytest tests/test_base.py::test_ssl_verification_option -v`
Run: `pytest tests/test_base.py::test_ssl_verification_disabled -v`
Run: `pytest tests/test_base.py::test_ssl_verification_default -v`
Expected: PASS

**Step 7: Run all tests**

Run: `pytest tests/test_base.py -v`
Expected: All tests pass

**Step 8: Commit**

```bash
git add lexilux/_base.py tests/test_base.py
git commit -m "feat(I02): add SSL verification options

Allow users to configure SSL verification:
- verify_ssl=True (default): Use system certificates
- verify_ssl=False: Disable SSL verification (not recommended)
- verify_ssl='/path/to/ca.crt': Use custom CA bundle

This enables compatibility with corporate proxies and custom CA setups."
```

---

## Final Verification

### Task 15: Full Test Suite and Coverage Check

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Verify coverage target**

Run: `pytest --cov=lexilux --cov-report=term-missing --cov-report=html`
Expected: Coverage >= 85%

**Step 3: Run linting**

Run: `make lint`
Expected: No linting errors

**Step 4: Run formatting check**

Run: `make format-check`
Expected: No formatting needed

**Step 5: Run benchmarks**

Run: `pytest benchmarks/ --benchmark-only --benchmark-autosave`
Expected: Benchmarks complete successfully

**Step 6: Verify client.py size**

Run: `wc -l lexilux/chat/client.py`
Expected: <= 500 lines (after refactoring)

**Step 7: Update documentation**

Update `README.md` with new features:
- Rate limiting (`rate_limit` parameter)
- SSL verification (`verify_ssl` parameter)
- Improved performance (caching, connection pooling)

Update `CHANGELOG.md`:

```markdown
## [2.5.0] - 2026-02-XX

### Added
- Client-side rate limiting with `rate_limit` parameter
- ModelRegistry LRU cache for improved performance
- SSL certificate verification options via `verify_ssl` parameter
- Performance benchmarking suite with pytest-benchmark
- Comprehensive concurrent safety tests

### Changed
- Extracted streaming logic to `lexilux/chat/streaming.py`
- Extracted continuation logic to `lexilux/chat/continue.py`
- Extracted validation logic to `lexilux/chat/validation.py`
- Fixed httpx connection limits for proper keep-alive
- All code comments standardized to English

### Fixed
- Connection leak in streaming iterators (now uses context manager)
- Input validation with clear error messages
- Test coverage increased to 85%+

### Performance
- Model spec lookups now cached (128 entries LRU)
- Connection pool properly reuses HTTP connections
- Benchmarks established for performance regression detection
```

**Step 8: Final commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs(I02): update documentation for v2.5.0

Document new features, improvements, and breaking changes.
Update changelog with all I02@A02 improvements."
```

---

## Success Criteria Checklist

Before completing, verify:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Coverage >= 85% (`pytest --cov=lexilux --cov-report=term-missing`)
- [ ] No linting errors (`make lint`)
- [ ] No formatting issues (`make format-check`)
- [ ] Benchmarks run successfully
- [ ] `chat/client.py` <= 500 lines
- [ ] All comments in English
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

**Total Estimated Time**: ~13.5 days
**Files Modified**: ~15 files
**Files Created**: ~10 files
**Test Files**: ~8 new test files
**Lines of Code**: ~2000 lines added/modified
