# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management. The Makefile automatically uses `uv` if available, otherwise falls back to `pip`.

```bash
# Install for active development (recommended)
make dev-install

# Alternative: Create venv with dependencies only (no package install)
make setup-venv

# Run unit tests (excludes integration tests)
make test

# Run tests with coverage
make test-cov

# Run integration tests (requires external services/API keys)
make test-integration

# Run linting
make lint

# Format code
make format

# Run all checks (lint + format check + tests)
make check

# Build documentation
make docs
```

### Running Single Tests

```bash
# Run a specific test file
uv run pytest tests/test_chat.py -v

# Run a specific test function
uv run pytest tests/test_chat.py::test_chat_basic -v

# Run with markers (e.g., integration tests)
uv run pytest tests/ -v -m integration
```

## Architecture Overview

Lexilux is a unified LLM API client library with a modular architecture. The codebase follows a clear separation of concerns:

### Core Components

1. **`lexilux/usage.py`**: Base classes for all API responses
   - `Usage`: Unified usage statistics (input_tokens, output_tokens, total_tokens, details)
   - `ResultBase`: Base class for all API results, ensures `.usage` and `.raw` are always available

2. **`lexilux/base.py`**: Base API client (v2.2.0+)
   - `BaseAPIClient`: Provides common HTTP functionality to all clients
   - Connection pooling with configurable pool sizes (default: 10 connections)
   - Automatic retry logic with exponential backoff
   - Separate connect/read timeout configuration
   - Request logging and timing
   - Authentication handling
   - Error response parsing and exception mapping

3. **`lexilux/exceptions.py`**: Exception hierarchy (v2.2.0+)
   - `LexiluxError`: Base exception with `code`, `message`, `retryable` properties
   - `AuthenticationError`: 401 errors (not retryable)
   - `RateLimitError`: 429 errors (retryable)
   - `TimeoutError`: Request timeouts (retryable)
   - `ConnectionError`: Connection failures (retryable)
   - `ValidationError`: 400 errors (not retryable)
   - `NotFoundError`: 404 errors (not retryable)
   - `ServerError`: 5xx errors (retryable)
   - `InvalidRequestError`: Alias for ValidationError
   - `ConfigurationError`: Client configuration issues (not retryable)
   - `NetworkError`: Base class for network issues

4. **`lexilux/chat/`**: Chat completion API (main module, subdirectory structure)
   - `client.py`: Core `Chat` class with `__call__()` (non-streaming) and `stream()` methods
   - `history.py`: `ChatHistory` dataclass with utility functions for message manipulation
   - `continue_.py`: `ChatContinue` for conversation continuation with customizable strategy
   - `formatters.py`: `ChatHistoryFormatter` for converting between message formats
   - `params.py`: `ChatParams` dataclass for structured parameter configuration
   - `streaming.py`: `StreamingResult` and `StreamingIterator` for streaming responses
   - `exceptions.py`: Custom exceptions (`ChatStreamInterruptedError`, `ChatIncompleteResponseError`)
   - `models.py`: Data models for responses
   - `utils.py`: Utility functions for message normalization

5. **`lexilux/embed.py`**: Embedding API client
6. **`lexilux/rerank.py`**: Rerank API client with multiple modes (openai, dashscope)
7. **`lexilux/tokenizer.py`**: Tokenizer client using HuggingFace transformers (optional dependency)

### Key Design Patterns

- **Function-like API**: All clients are callable (e.g., `chat("hello")` instead of `chat.create(messages=...)`)
- **Unified Usage Tracking**: Every API result includes a `Usage` object with consistent fields
- **Flexible Input**: Messages can be strings, lists of strings, or lists of dicts
- **Streaming Support**: Chat supports both non-streaming (`__call__`) and streaming (`stream()`) responses
- **OpenAI-Compatible**: Works with any OpenAI-compatible API
- **Connection Pooling** (v2.2.0+): All clients use HTTP connection pooling for better performance
- **Automatic Retry** (v2.2.0+): Configurable retry logic with exponential backoff for transient failures
- **Exception Hierarchy** (v2.2.0+): Structured exception system with error codes and retryable flags
- **Request Logging** (v2.2.0+): Built-in logging for debugging and monitoring (disabled by default)

### Chat Module Details

The chat module is the most complex part of the codebase:

- **Message Normalization**: The `normalize_messages()` utility in `chat/utils.py` converts various input formats (str, list of str, list of dict) into the standardized OpenAI message format
- **History Immutability**: `ChatHistory` is immutable (frozen dataclass). Modifications return new instances via `replace()` method.
- **Continue Strategy**: `ChatContinue` supports two strategies for continuing conversations:
  - `"history"`: Append new messages to history (default, preserves context)
  - `"last"`: Only include the last N messages (reduces token usage)
- **Streaming**: The `stream()` method yields `ChatStreamChunk` objects with `delta`, `done`, `finish_reason`, and `usage` fields

### Parameter Handling

Chat supports two parameter passing styles (for backward compatibility):

1. **Individual parameters** (backward compatible): `chat("hello", temperature=0.5, max_tokens=100)`
2. **ChatParams dataclass**: `chat("hello", params=ChatParams(temperature=0.5))`

When both are provided, individual parameters override the ChatParams values. The `extra` parameter allows passing custom provider-specific parameters.

### Testing Structure

- Unit tests are in `tests/test_*.py` and run with `make test`
- Integration tests (marked with `@pytest.mark.integration`) require external services
- Test fixtures are in `tests/conftest.py`
- Integration test endpoints are configured in `tests/test_endpoints.json`

## Important Notes

- **Python Version**: Supports Python 3.8-3.14 (specified in pyproject.toml)
- **Dependencies**: Core requires only `requests>=2.28.0`. Tokenizer support is optional (`[tokenizer]` extra)
- **Line Length**: Code is formatted to 100 characters (ruff configuration)
- **Type Hints**: The codebase uses type hints extensively but mypy is configured permissively
- **Proxy Configuration**: All API clients support proxies via the `proxies` parameter. If `None`, uses environment variables (HTTP_PROXY, HTTPS_PROXY). Pass `{}` to disable proxies.

## Common Tasks

### Configuring Retry Logic

All clients support automatic retry with exponential backoff for transient failures:

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    max_retries=3,  # Automatically retry on 429, 500, 502, 503, 504
)
```

Retry behavior:
- Retries on status codes: 429, 500, 502, 503, 504
- Exponential backoff: 0.1s, 0.2s, 0.4s...
- Check `e.retryable` property to determine if an error is retryable

### Enabling Request Logging

Enable logging to debug requests and monitor performance:

```python
import logging

logging.basicConfig(level=logging.INFO)

from lexilux import Chat
chat = Chat(base_url="...", api_key="...")
result = chat("Hello")
# Logs: "Request completed in 0.52s with status 200: https://..."
```

### Configuring Connection Pooling

Adjust connection pool size for high-concurrency scenarios:

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    pool_connections=20,  # Default: 10
    pool_maxsize=20,      # Default: 10
)
```

### Handling Errors

Use the exception hierarchy for robust error handling:

```python
from lexilux import Chat, LexiluxError, AuthenticationError, RateLimitError

chat = Chat(base_url="...", api_key="...")

try:
    result = chat("Hello")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
    print(f"Error code: {e.code}")  # "authentication_failed"
    print(f"Can retry: {e.retryable}")  # False
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
    print(f"Can retry: {e.retryable}")  # True
except LexiluxError as e:
    print(f"Error: {e.code} - {e.message}")
    if e.retryable:
        # Implement retry logic
        pass
```

### Adding a New Chat Parameter

1. Add the parameter to `ChatParams` dataclass in `lexilux/chat/params.py`
2. Add handling in `Chat.__call__()` and `Chat.stream()` methods in `lexilux/chat/client.py`
3. Update docstrings and tests

### Adding Support for a New Provider

1. Check if the provider is OpenAI-compatible (most are)
2. If compatible, just set the correct `base_url` when instantiating clients
3. If not compatible, you may need to add custom parameter handling via the `extra` parameter or create a new mode (see `Rerank` for examples of different modes)
