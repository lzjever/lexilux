# Lexilux I02@A02 Design Document

**Date**: 2026-02-08
**Author**: Claude (World-Class Senior Engineer + Top Architect)
**Status**: Design Phase
**Target Release**: v2.5.0

---

## 0) Executive Summary

**Goal**: Address all issues identified in A02 code review (P0-P3) through a single major iteration, focusing on stability, performance, and code quality while maintaining the library's core responsibility as an LLM client library.

**Scope**: 3 functional packages, ~13.5 days of work, single feature branch `vk/9de9-lexilux-i02-a02`

**Key Principles**:
- KISS: Keep solutions simple and focused
- DRY: Don't repeat yourself
- SOLID: Single responsibility, open/closed, dependency inversion
- YAGNI: You aren't gonna need it (skip request batching)

---

## 1) Architecture Overview

### 1.1 Current State

```
lexilux/
├── _base.py               # HTTP client base (821 lines)
├── chat/
│   ├── client.py          # Main client (1100 lines) - TOO LARGE
│   ├── _request.py        # Request building
│   ├── streaming.py       # Streaming iterator
│   ├── history.py         # Conversation history
│   └── ...
├── exceptions.py          # Exception hierarchy
├── registry/
│   └── registry.py        # Model registry (no caching)
└── ...
```

### 1.2 Target State

```
lexilux/
├── _base.py               # Enhanced with SSL options
├── chat/
│   ├── client.py          # Core client (~400 lines)
│   ├── streaming.py       # Streaming logic (extracted)
│   ├── continue.py        # Conversation continuation (extracted)
│   ├── validation.py      # Parameter validation (extracted)
│   ├── _request.py        # Request building (unchanged)
│   └── ...
├── exceptions.py          # Enhanced with validation errors
├── registry/
│   └── registry.py        # With LRU cache
├── _rate_limit.py         # NEW: Rate limiter
└── ...
```

---

## 2) Package 1: Stability (5.5 days)

### 2.1 P0-1: Test Coverage 68% → 85% (Risk-Oriented)

**Strategy**: Focus on high-risk code paths, not blanket coverage.

**Priority Areas**:

1. **Core Execution Paths**
   - Request building: `chat/_request.py:build_chat_payload()`
   - Response parsing: `chat/_request.py:parse_chat_completion_response()`
   - Error handling: `_base.py:_handle_response_error()`

2. **Boundary Conditions**
   - Empty/null inputs (empty messages, None parameters)
   - Extreme values (max_tokens=0, temperature=2.0)
   - Large inputs (100+ messages, long content)

3. **Exception Flows**
   - Network errors (timeout, connection refused)
   - API errors (401, 429, 500)
   - Malformed responses

**Test Files to Create/Extend**:

```python
tests/chat/test_boundary_conditions.py
tests/chat/test_exception_paths.py
tests/test_base_edge_cases.py
tests/test_concurrent.py  # NEW: concurrent safety
```

**Tools**: pytest, pytest-asyncio, hypothesis (optional for property testing)

---

### 2.2 P0-2: Connection Leak Fix

**Problem**: Streaming iterators may leak connections when interrupted.

**Solution**: Use context managers for guaranteed cleanup.

**Implementation**:

```python
# lexilux/_base.py
from contextlib import contextmanager

@contextmanager
def _streaming_request_context(self, endpoint: str, payload: dict):
    """Context manager for streaming requests with guaranteed cleanup."""
    url = f"{self.base_url}/{endpoint}"
    response = self._session.post(
        url,
        json=payload,
        stream=True,
        timeout=self.timeout,
        headers=self.headers,
    )
    try:
        yield response
    finally:
        response.close()

# Usage in chat/client.py
def stream(self, messages, **params):
    payload = self._build_payload(messages, stream=True, **params)
    with self._streaming_request_context("chat/completions", payload) as response:
        for line in response.iter_lines():
            # ... process lines
```

---

### 2.3 P0-3: httpx Connection Parameters

**Problem**: `max_keepalive_connections=0` prevents connection reuse.

**Fix**:

```python
# lexilux/_base.py (line ~535)
# BEFORE:
limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)

# AFTER:
limits=httpx.Limits(
    max_connections=10,
    max_keepalive_connections=5,
)
```

---

### 2.4 P1-1: Concurrent Testing

**Scope**: Verify thread safety and asyncio concurrency safety.

**Test Scenarios**:

1. **Thread Safety** (for sync clients)
   - Multiple threads sharing same Chat instance
   - Connection pool under concurrent load
   - Session object thread safety

2. **Asyncio Concurrency**
   - Multiple concurrent async requests
   - Async connection pool behavior
   - Streaming under concurrent load

**Implementation**:

```python
# tests/test_concurrent.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

def test_concurrent_sync_requests():
    """Test multiple threads using same Chat instance."""
    chat = Chat(base_url="...", api_key="test", model="gpt-4")

    def make_request():
        chat("Hello")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        for f in futures:
            f.result()  # Should not raise

@pytest.mark.asyncio
async def test_concurrent_async_requests():
    """Test multiple concurrent async requests."""
    chat = Chat(base_url="...", api_key="test", model="gpt-4")

    async def make_request():
        await chat("Hello")

    await asyncio.gather(*[make_request() for _ in range(50)])
```

---

## 3) Package 2: Performance (4 days)

### 3.1 P1-2: Performance Benchmarking

**Tool**: `pytest-benchmark`

**Setup**:

```toml
# pyproject.toml
[project.optional-dependencies]
benchmark = [
    "pytest-benchmark>=4.0.0",
]
```

**Benchmark Files**:

```python
# benchmarks/test_chat_performance.py
import pytest
from lexilux import Chat

@pytest.mark.benchmark
def test_chat_latency(benchmark):
    """Benchmark single chat request latency."""
    chat = Chat(base_url="...", api_key="...", model="gpt-4")
    result = benchmark(chat, "Hello, world!")
    assert result.content

@pytest.mark.benchmark
def test_chat_throughput(benchmark):
    """Benchmark chat throughput (requests per second)."""
    chat = Chat(base_url="...", api_key="...", model="gpt-4")

    def make_requests():
        for _ in range(10):
            chat("test")

    benchmark(make_requests)
```

**Metrics to Track**:
- p50, p95, p99 latency
- Requests per second
- Memory usage

---

### 3.2 P2-1: Client-Side Rate Limiting

**Tool**: `aiolimiter`

**Implementation**:

```python
# lexilux/_rate_limit.py (NEW FILE)
from aiolimiter import AsyncLimiter

class RateLimiter:
    """Client-side rate limiter to prevent API quota exhaustion."""

    def __init__(self, rate_limit: int = 60, time_period: float = 1.0):
        """
        Args:
            rate_limit: Max requests allowed
            time_period: Time window in seconds
        """
        self._limiter = AsyncLimiter(rate_limit, time_period)

    async def acquire(self):
        """Acquire permission to make a request."""
        await self._limiter.acquire()

# Integration in chat/client.py
from lexilux._rate_limit import RateLimiter

class Chat(BaseAPIClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str,
        rate_limit: int | None = None,  # NEW parameter
        # ... other params
    ):
        # ... existing init
        self._rate_limiter = RateLimiter(rate_limit or 60) if rate_limit else None

    async def _amake_request(self, endpoint: str, payload: dict):
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        return await super()._amake_request(endpoint, payload)
```

**Dependencies**:

```toml
[project.optional-dependencies]
rate-limit = [
    "aiolimiter>=1.1.0",
]
```

---

### 3.3 P2-3: ModelRegistry LRU Cache

**Tool**: `functools.lru_cache` (standard library)

**Implementation**:

```python
# lexilux/registry/registry.py
from functools import lru_cache

class ModelRegistry:
    # ... existing code

    @lru_cache(maxsize=128)
    def _get_model_spec_cached(self, model_id: str) -> ModelSpec:
        """Cached version of model spec lookup."""
        return self._get_model_spec(model_id)

    def get_model_spec(self, model_id: str) -> ModelSpec:
        """Get model spec with caching."""
        try:
            return self._get_model_spec_cached(model_id)
        except UnknownModelError:
            # Fallback to uncached for warning logic
            return self._get_model_spec(model_id)
```

**Benefits**:
- Eliminates redundant file I/O for repeated model lookups
- Zero additional dependencies
- Automatic cache management

---

## 4) Package 3: Code Quality (4 days)

### 4.1 P1-4: chat/client.py Refactoring

**Strategy**: Extract by functional responsibility.

**New Structure**:

```
chat/
├── client.py          # Core coordination (~400 lines)
├── streaming.py       # Streaming logic (~200 lines)
├── continue.py        # Conversation continuation (~150 lines)
└── validation.py      # Parameter validation (~150 lines)
```

**Detailed Breakdown**:

**client.py** (Core):
- `__init__`: Initialization
- `__call__`: Main entry point
- `complete`: High-level complete API
- Request/response coordination
- Shared sync/async logic

**streaming.py** (Extracted):
- `stream()` method
- `astream()` method
- `StreamingIterator` class
- SSE parsing integration

**continue.py** (Extracted):
- `continue_conversation()` method
- `continue_stream()` method
- Working history management
- Conversation state logic

**validation.py** (Extracted):
- `ChatParams` validation
- Input sanitization
- Custom validation exceptions

---

### 4.2 P2-2: Lightweight Input Validation

**Scope**: Basic parameter validation, no external dependencies.

**Implementation**:

```python
# lexilux/chat/validation.py (NEW FILE)

class ValidationError(LexiluxError):
    """Raised when input validation fails."""
    code = "validation_error"
    retryable = False

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
        raise ValidationError(f"n must be at least 1, got {params.n}")

def validate_messages(messages: list | str) -> list:
    """
    Validate and normalize messages input.

    Args:
        messages: String message or list of message dicts

    Returns:
        Normalized list of messages

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if not messages:
        raise ValidationError("messages cannot be empty")

    if not isinstance(messages, list):
        raise ValidationError(f"messages must be list or str, got {type(messages)}")

    return messages

# Integration in chat/client.py
from lexilux.chat.validation import validate_chat_params, validate_messages

def __call__(self, messages, **params):
    normalized = validate_messages(messages)
    chat_params = ChatParams(**params)
    validate_chat_params(chat_params)
    # ... continue with request
```

---

### 4.3 P3-1: English Comments

**Scope**: Replace all Chinese comments with English equivalents.

**Files to Update**:
- `lexilux/chat/_request.py:279` (known location)
- Any other files with non-English comments

**Approach**:
1. Use grep to find all non-ASCII comment patterns
2. Translate to clear, concise English
3. Maintain technical accuracy

---

### 4.4 P3-2: SSL Certificate Options

**Scope**: Allow custom SSL verification without full certificate pinning.

**Implementation**:

```python
# lexilux/_base.py
class BaseAPIClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        verify_ssl: str | bool = True,  # NEW parameter
        # ... other params
    ):
        """
        Args:
            base_url: Base URL for API
            api_key: API key for authentication
            timeout_s: Request timeout in seconds
            verify_ssl: SSL verification (True=default, False=disable, or path to CA bundle)
        """
        # ... existing code
        self._verify_ssl = verify_ssl

    def _make_request(self, endpoint: str, payload: dict) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        return self._session.post(
            url,
            json=payload,
            timeout=self.timeout,
            headers=self.headers,
            proxies=self.proxies,
            verify=self._verify_ssl,  # Use verify_ssl parameter
        )
```

**Usage Examples**:

```python
# Default (system certificates)
chat = Chat(base_url="...", api_key="...")

# Custom CA bundle
chat = Chat(base_url="...", api_key="...", verify_ssl="/path/to/ca-bundle.crt")

# Disable SSL (not recommended)
chat = Chat(base_url="...", api_key="...", verify_ssl=False)
```

---

## 5) Dependencies Summary

**New Production Dependencies**:

| Package | Version | Purpose | Optional? |
|---------|---------|---------|-----------|
| aiolimiter | >=1.1.0 | Rate limiting | Yes (rate-limit extra) |

**New Development Dependencies**:

| Package | Version | Purpose |
|---------|---------|---------|
| pytest-benchmark | >=4.0.0 | Performance benchmarking |
| pytest-asyncio | >=0.21.0 | Async testing (may already exist) |

**Removed from Scope**:
- ~~structlog~~ (not library's responsibility)
- ~~opentelemetry~~ (not library's responsibility)
- ~~pydantic~~ (using lightweight validation)
- ~~certvalidator~~ (using built-in SSL options)

---

## 6) Success Criteria

**Package 1: Stability**
- [ ] Test coverage >= 85% (measured by pytest-cov)
- [ ] All concurrent tests pass (50+ concurrent operations)
- [ ] No connection leaks (verified with resource monitoring)
- [ ] httpx connection reuse verified (via logging/debug)

**Package 2: Performance**
- [ ] Baseline metrics established (p50/p95/p99 latency)
- [ ] Rate limiting functional (respects configured limits)
- [ ] ModelRegistry cache effective (cache hit rate > 90% for repeated queries)

**Package 3: Code Quality**
- [ ] client.py <= 500 lines
- [ ] All comments in English
- [ ] Validation prevents invalid API calls
- [ ] SSL options configurable

**Overall**
- [ ] All existing tests pass
- [ ] No new linting errors
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## 7) Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes during refactoring | High | Comprehensive test suite before refactoring |
| Performance regression | Medium | Benchmark before/after each change |
| Dependency conflicts | Low | All new deps are optional or well-maintained |
| Scope creep | Medium | YAGNI principle applied (e.g., skipped batching) |

---

## 8) Implementation Order

**Phase 1: Foundation** (Package 1)
1. Fix P0-3 (httpx params) - 0.1 day
2. Add concurrent tests - 2 days
3. Increase test coverage - 2 days
4. Fix connection leaks - 0.5 days
5. Verify all stability improvements

**Phase 2: Optimization** (Package 2)
1. Setup pytest-benchmark - 0.5 days
2. Establish baselines - 0.5 days
3. Implement rate limiting - 1 day
4. Add ModelRegistry cache - 0.5 days
5. Verify performance improvements

**Phase 3: Refinement** (Package 3)
1. Refactor client.py - 1 day
2. Add validation module - 1 day
3. English comments - 0.5 days
4. SSL options - 0.5 days
5. Final verification

---

## 9) Next Steps

After design approval:

1. Use `superpowers:writing-plans` to create detailed implementation plan
2. Use `superpowers:executing-plans` to execute implementation
3. Use `superpowers:verification-before-completion` before final commit

---

**Document Status**: Ready for implementation planning phase
