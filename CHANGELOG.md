# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-01-15

### 🎯 Quality & Infrastructure Improvements

This release focuses on code quality, robustness, and developer experience improvements without breaking changes.

### Added

#### Connection Pooling & Performance
- **Connection Pooling**: All API clients now use HTTP connection pooling for better performance under high concurrency
  - Configurable via `pool_connections` and `pool_maxsize` parameters (default: 10 each)
  - Reduces connection overhead for repeated requests
  - Improves performance in high-throughput scenarios

#### Automatic Retry Logic
- **Retry with Exponential Backoff**: Automatic retry for transient failures
  - Configurable via `max_retries` parameter (default: 0, disabled)
  - Retries on status codes: 429, 500, 502, 503, 504
  - Exponential backoff: 0.1s, 0.2s, 0.4s...
  - Helps recover from temporary network issues

#### Enhanced Timeout Configuration
- **Separate Timeouts**: Fine-grained timeout control for connection and read phases
  - `connect_timeout_s`: Connection establishment timeout (default: from `timeout_s`)
  - `read_timeout_s`: Data read timeout (default: from `timeout_s`)
  - Legacy `timeout_s` parameter still supported for backward compatibility
  - Allows different timeouts for connect vs read operations

#### Unified Exception Hierarchy
- **Custom Exception System**: Complete exception hierarchy with error codes and retryable flags
  - `LexiluxError` - Base exception class for all Lexilux errors
  - `AuthenticationError` - Authentication/authorization failures (401, not retryable)
  - `RateLimitError` - Rate limit exceeded (429, retryable)
  - `TimeoutError` - Request timeouts (retryable)
  - `ConnectionError` - Connection failures (retryable)
  - `ValidationError` - Invalid input (400, not retryable)
  - `NotFoundError` - Resource not found (404, not retryable)
  - `ServerError` - Internal server errors (5xx, retryable)
  - `InvalidRequestError` - Alias for ValidationError
  - `ConfigurationError` - Client configuration issues (not retryable)
  - `NetworkError` - Base class for network issues
  - All exceptions have `code`, `message`, and `retryable` properties

#### Logging & Monitoring
- **Request Logging**: Comprehensive logging for debugging and monitoring
  - Logs request start, completion, timing, and errors
  - Uses appropriate log levels (DEBUG, INFO, WARNING, ERROR)
  - Enable with: `import logging; logging.basicConfig(level=logging.INFO)`
  - Helps with debugging and performance monitoring

#### BaseAPIClient Architecture
- **New Base Class**: `BaseAPIClient` provides common HTTP functionality to all clients
  - Session management with connection pooling
  - Retry logic with exponential backoff
  - Configurable timeouts (connect/read)
  - Authentication handling
  - Error response parsing and exception mapping
  - Request logging and timing

#### Documentation
- **CONTRIBUTING.md**: Comprehensive contribution guidelines
  - Code style guidelines (PEP 8, type hints, docstrings)
  - Commit message format (Conventional Commits)
  - PR workflow and checklist
  - Bug report and feature request templates
  - Coverage goals and test structure examples
- **docs/source/troubleshooting.rst**: Troubleshooting guide for common issues
  - Installation issues (module not found, version conflicts)
  - Connection issues (timeout, connection refused)
  - Authentication issues (401, 403)
  - Rate limiting (429)
  - Streaming issues
  - Performance issues
  - Debugging techniques
  - Common errors reference table
- **TESTING.md**: Testing documentation with coverage goals and guidelines
- **Updated Examples**: `error_handling_demo.py` updated to use new exception hierarchy

#### CI/CD Improvements
- **Multi-Version Testing**: CI now tests across Python 3.8-3.14 in separate jobs
- **Security Scanning**: Automated vulnerability detection
  - pip-audit for dependency vulnerabilities
  - bandit for code security issues
  - Runs daily and on every push/PR
- **Pre-commit Hooks**: Code quality checks before commits
  - ruff lint and format
  - trailing whitespace and file ending fixes
  - YAML syntax checking
- **Coverage Threshold**: Minimum 60% code coverage enforced in CI
- **Separate Lint Job**: Lint and format checks run in parallel with tests

### Changed

#### Chat Client Improvements
- **BaseAPIClient Integration**: Chat now inherits from `BaseAPIClient` for consistent HTTP behavior
  - All HTTP requests now use connection pooling
  - Network errors raise custom exceptions instead of raw requests exceptions
  - Consistent timeout and retry behavior across all clients

#### CI/CD Architecture
- **Enhanced Pipeline**: Lint and format checks run in parallel with tests
  - Coverage uploaded only from Python 3.14 to save CI resources
  - Separate security workflow for dependency and code scanning

### Fixed

#### Bug Fixes
- **ChatHistory Deep Copy Protection**: Added deep copy to prevent external modifications to internal state
  - Messages list is deep copied during initialization
  - Prevents state pollution from external code modifying original input
  - Ensures history immutability
- **Error Message Extraction**: API errors now extract and include detailed error messages from JSON response bodies
  - Supports OpenAI-style error format (`{"error": {"message": "..."}}`)
  - Falls back to generic message if parsing fails
- **Timeout Handling**: Network timeouts now raise `TimeoutError` instead of generic `requests.exceptions.Timeout`
- **Backward Compatibility**: Added `timeout_s` property to Chat for backward compatibility with tuple timeout configuration
- **Test Mocks**: Fixed test mocks to work with BaseAPIClient architecture

#### Code Quality
- Fixed all ruff linting and formatting issues
- Removed unused imports and variables
- Corrected import order

### Security

- All dependency vulnerabilities now scanned in CI pipeline using pip-audit
- Code security linted with bandit for common security issues
- Security scan runs daily and on every push/PR
- Found issues are acceptable for library context (non-critical use cases)

### Migration Guide

#### Enabling Retry Logic

```python
from lexilux import Chat

# Enable automatic retry with exponential backoff
chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    max_retries=3,  # Automatically retry on transient failures
)

# Manual retry using retryable flag
from lexilux import LexiluxError
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        result = chat("Hello, world!")
        break
    except LexiluxError as e:
        if e.retryable and attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

#### Using New Exceptions

```python
from lexilux import (
    Chat,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    LexiluxError,
)

try:
    result = chat("Hello, world!")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
    print(f"Error code: {e.code}")  # "authentication_failed"
    print(f"Can retry: {e.retryable}")  # False
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
    print(f"Error code: {e.code}")  # "rate_limit_exceeded"
    print(f"Can retry: {e.retryable}")  # True
except LexiluxError as e:
    print(f"Error: {e.code} - {e.message}")
```

#### Enabling Logging

```python
import logging

# Enable INFO level logging
logging.basicConfig(level=logging.INFO)

from lexilux import Chat
chat = Chat(base_url="...", api_key="...")
result = chat("Hello")
# Logs: "Request completed in 0.52s with status 200: https://..."
```

#### Configuring Connection Pooling

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    pool_connections=20,  # Increase for high concurrency
    pool_maxsize=20,
)
```

#### Separate Timeouts

```python
from lexilux import Chat

# New API: separate connect and read timeouts
chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    connect_timeout_s=5,   # Connection timeout
    read_timeout_s=30,     # Read timeout
)

# Old API still works
chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    timeout_s=30,  # Used for both connect and read
)
```

### Developer Experience

- Better error messages with error codes
- Automatic retry reduces manual error handling
- Logging helps with debugging
- Comprehensive documentation for troubleshooting
- Clear contribution guidelines
- Automated quality checks (pre-commit, CI)

### Performance

- Connection pooling reduces overhead for repeated requests
- Retry logic with exponential backoff improves reliability
- Request timing via logging helps identify bottlenecks

## [Unreleased]

### Added
- **Connection Pooling**: All API clients now use connection pooling for better performance under high concurrency
  - Configurable via `pool_connections` and `pool_maxsize` parameters
  - Reduces connection overhead for repeated requests
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
  - Configurable via `max_retries` parameter (default: 0, disabled)
  - Retries on status codes: 429, 500, 502, 503, 504
  - Exponential backoff: 0.1s, 0.2s, 0.4s...
- **Timeout Configuration**: Separate `connect_timeout_s` and `read_timeout_s` parameters for fine-grained timeout control
  - Legacy `timeout_s` parameter still supported for backward compatibility
- **Unified Exception Hierarchy**: Complete exception system with error codes and retryable flags
  - `LexiluxError` - Base exception class for all Lexilux errors
  - `AuthenticationError` - Authentication/authorization failures (401, not retryable)
  - `RateLimitError` - Rate limit exceeded (429, retryable)
  - `TimeoutError` - Request timeouts (retryable)
  - `ConnectionError` - Connection failures (retryable)
  - `ValidationError` - Invalid input (400, not retryable)
  - `NotFoundError` - Resource not found (404, not retryable)
  - `ServerError` - Internal server errors (5xx, retryable)
  - `InvalidRequestError` - Alias for ValidationError
  - `ConfigurationError` - Client configuration issues (not retryable)
  - `NetworkError` - Base class for network issues
- **Logging Support**: Request logging for debugging and monitoring
  - Logs request start, completion, timing, and errors
  - Uses appropriate log levels (DEBUG, INFO, WARNING, ERROR)
  - Enable with: `import logging; logging.basicConfig(level=logging.INFO)`
- **BaseAPIClient**: New base class providing common HTTP functionality to all clients
  - Session management with connection pooling
  - Retry logic with exponential backoff
  - Configurable timeouts (connect/read)
  - Authentication handling
  - Error response parsing and exception mapping
  - Request logging and timing
- **Documentation**:
  - `CONTRIBUTING.md` - Comprehensive contribution guidelines with code style, testing, and PR templates
  - `docs/source/troubleshooting.rst` - Troubleshooting guide for common issues
  - `TESTING.md` - Testing documentation with coverage goals and guidelines
- **Security Scanning**: CI workflow with pip-audit and bandit for vulnerability detection
- **Multi-Version Testing**: CI now tests across Python 3.8-3.14 in separate jobs
- **Pre-commit Hooks**: Code quality checks before commits (ruff lint and format)
- **Coverage Threshold**: Minimum 60% code coverage enforced in CI
- **Updated Examples**: `error_handling_demo.py` updated to use new exception hierarchy

### Changed
- **Chat**: Now inherits from `BaseAPIClient` for consistent HTTP behavior
  - All HTTP requests now use connection pooling
  - Network errors raise custom exceptions instead of raw requests exceptions
- **CI/CD**: Enhanced with separate lint job, multi-version testing matrix, and security scanning
  - Lint and format checks run in parallel with tests
  - Coverage uploaded only from Python 3.14 to save resources

### Fixed
- **ChatHistory**: Added deep copy protection to prevent external modifications to internal state
  - Messages list is deep copied during initialization
  - Prevents state pollution from external code modifying original input
- **Error Messages**: API errors now extract and include detailed error messages from JSON response bodies
  - Supports OpenAI-style error format (`{"error": {"message": "..."}}`)
  - Falls back to generic message if parsing fails
- **Timeout Handling**: Network timeouts now raise `TimeoutError` instead of generic `requests.exceptions.Timeout`
- **Backward Compatibility**: Added `timeout_s` property to Chat for backward compatibility with tuple timeout

### Security
- All dependency vulnerabilities now scanned in CI pipeline using pip-audit
- Code security linted with bandit for common security issues
- Security scan runs daily and on every push/PR

## [2.1.0] - 2026-01-10

### 🎯 API Improvements: History Immutability & Customizable Continue Strategy

This minor version update introduces important API improvements focusing on immutability, clarity, and customization capabilities.

### Changed

- **History Immutability**: All methods that receive a `history` parameter now create a clone internally and never modify the original history object. This ensures:
  - No unexpected side effects
  - Thread-safe operations (multiple threads can use the same history)
  - Functional programming principles
  - Better predictability

- **`chat.complete()` and `chat.complete_stream()` history parameter**: Now **optional** instead of required. If `None`, a new `ChatHistory` instance is created internally. This simplifies single-turn complete requests:
  ```python
  # Before (v2.0.0)
  history = ChatHistory()
  result = chat.complete("Write JSON", history=history)
  
  # After (v2.1.0)
  result = chat.complete("Write JSON")  # No history needed for single-turn
  ```

- **API Clarity**: Updated docstrings to clearly distinguish between:
  - `chat()` / `chat.stream()` → Single response (may be truncated)
  - `chat.complete()` / `chat.complete_stream()` → Complete response (guaranteed)

### Added

- **Customizable Continue Strategy**: Enhanced `chat.complete()` and `ChatContinue.continue_request()` with extensive customization options:
  - **Custom continue prompt**: Support for function-based prompts: `continue_prompt: str | Callable`
  - **Progress tracking**: `on_progress` callback for monitoring continuation progress
  - **Request delay control**: `continue_delay` parameter (fixed or random range)
  - **Error handling strategies**: `on_error` and `on_error_callback` for flexible error handling
  - **Helper method**: `ChatContinue.needs_continue(result)` to check if continuation is needed

- **Enhanced `ChatContinue.continue_request()` and `continue_request_stream()`**:
  - Support for all customization options (progress, delay, error handling)
  - History immutability (clones internally)
  - Better error handling and recovery

### Removed

- **`Chat.continue_if_needed()`**: Removed in favor of `chat.complete()` which provides the same functionality with better API clarity.
- **`Chat.continue_if_needed_stream()`**: Removed in favor of `chat.complete_stream()`.

### Migration Guide

#### Using `continue_if_needed()` → Use `complete()` instead

```python
# Before (v2.0.0)
history = ChatHistory()
result = chat("Write JSON", history=history, max_tokens=100)
if result.finish_reason == "length":
    full_result = chat.continue_if_needed(result, history=history)

# After (v2.1.0)
result = chat.complete("Write JSON", max_tokens=100)  # Automatically handles truncation
```

#### History Immutability

```python
# Before (v2.0.0) - history was modified
history = ChatHistory()
result = chat("Hello", history=history)
# history now contains: [user: "Hello", assistant: result.text]

# After (v2.1.0) - history is immutable, manual update needed for multi-turn
history = ChatHistory()
result = chat("Hello", history=history)
# history is unchanged, manually update if needed:
history.add_user("Hello")
history.append_result(result)
```

#### Custom Continue Strategy

```python
# New in v2.1.0: Customizable continue behavior
def on_progress(count, max_count, current, all_results):
    print(f"🔄 Continuing {count}/{max_count}...")

def smart_prompt(count, max_count, current_text, original_prompt):
    return f"Please continue (attempt {count}/{max_count})"

result = chat.complete(
    "Write a long JSON",
    max_tokens=100,
    continue_prompt=smart_prompt,
    on_progress=on_progress,
    continue_delay=(1.0, 2.0),  # Random delay 1-2 seconds
    on_error="return_partial",  # Return partial on error
)
```

## [2.0.0] - 2026-01-07

### 🚀 Major Architecture Overhaul: Explicit History Management

This is a **major version update** with significant architectural changes. The core design philosophy has shifted from implicit to explicit history management, providing better control, predictability, and consistency.

### Changed

#### Breaking Changes

- **Removed `auto_history` parameter**: The `auto_history` parameter has been completely removed from `Chat.__init__()`. History management is now always explicit.
  - **Migration**: Create a `ChatHistory` instance and pass it explicitly to all methods:
    ```python
    # Before (v0.5.x)
    chat = Chat(..., auto_history=True)
    result = chat("Hello")
    history = chat.get_history()
    
    # After (v2.0.0)
    history = ChatHistory()
    result = chat("Hello", history=history)
    ```

- **All Chat methods now require explicit `history` parameter**: All methods that interact with history now accept an explicit `history: ChatHistory | None` parameter.
  - `Chat.__call__(messages, *, history=None, **params)`
  - `Chat.stream(messages, *, history=None, **params)`
  - `Chat.complete(messages, *, history: ChatHistory, **params)` (now required)
  - `Chat.continue_if_needed(result, *, history: ChatHistory, **params)` (now required)
  - `ChatContinue.continue_request(chat, last_result, *, history: ChatHistory, **params)` (now required)
  - `ChatContinue.continue_request_stream(chat, last_result, *, history: ChatHistory, **params)` (now required)

- **Removed history management methods from `Chat` class**:
  - `Chat.get_history()` - Use explicit `history` parameter instead
  - `Chat.clear_history()` - Use `history.clear()` instead
  - `Chat.clear_last_assistant_message()` - Use `history.remove_last()` instead

- **Simplified `ChatResult`**: `ChatResult` now only contains the result of a single LLM request, without any merged history information. This makes the API more predictable and easier to understand.

- **Unified Chat interface**: All Chat methods now only accept a single turn's message, with history being managed explicitly. This ensures consistent behavior between streaming and non-streaming modes.

### Added

- **Enhanced `ChatHistory` with `MutableSequence` protocol**: `ChatHistory` now implements Python's `collections.abc.MutableSequence` protocol, enabling array-like operations:
  - **Indexing**: `history[0]` - Get message by index
  - **Slicing**: `history[1:5]` - Get slice as new `ChatHistory` instance
  - **Iteration**: `for msg in history` - Iterate over messages
  - **Length**: `len(history)` - Get number of messages
  - **Membership**: `msg in history` - Check if message exists
  - **Assignment**: `history[0] = new_msg` - Replace message at index
  - **Deletion**: `del history[0]` - Remove message at index
  - **Insertion**: `history.insert(0, msg)` - Insert message at index

- **New `ChatHistory` methods**:
  - `clone()` - Create a deep copy of the history
  - `__add__(other)` - Merge two `ChatHistory` instances: `history1 + history2`
  - `add_system(content)` - Explicitly add or update system message
  - `remove_last()` - Remove the last message
  - `remove_at(index)` - Remove message at specific index
  - `replace_at(index, message)` - Replace message at specific index
  - `get_user_messages()` - Get all user message contents as a list
  - `get_assistant_messages()` - Get all assistant message contents as a list
  - `get_last_message()` - Get the last message dictionary
  - `get_last_user_message()` - Get the content of the last user message

- **Streaming complete functionality**:
  - `Chat.complete_stream(messages, *, history: ChatHistory, ...)` - Streaming version of `complete()` that ensures complete responses with real-time chunk streaming
  - Supports progress callbacks (`on_progress`, `on_continue_start`, `on_continue_end`) for monitoring continuation progress

- **Streaming continue functionality**:
  - `ChatContinue.continue_request_stream(chat, last_result, *, history: ChatHistory, ...)` - Stream continuation chunks in real-time
  - Automatically merges results from all continuation requests
  - Provides access to accumulated result via `iterator.result.to_chat_result()`

- **Convenience methods**:
  - `Chat.chat_with_history(history, message=None, **params)` - Convenience method for chat with history
  - `Chat.stream_with_history(history, message=None, **params)` - Convenience method for streaming with history

### Improved

- **Explicit history management**: All history operations are now explicit, making the API more predictable and easier to debug.
- **Consistency**: Streaming and non-streaming modes now have identical behavior regarding history management.
- **Type safety**: Better type hints and validation for history parameters.
- **Error handling**: More precise error messages when history is required but not provided.
- **Documentation**: Comprehensive documentation updates reflecting the new explicit history management approach.

### Fixed

- **Fixed `finish_reason` propagation in streaming responses**: Corrected issue where `finish_reason` could be incorrectly set to `None` in streaming responses, especially during continuation requests.
- **Fixed history update timing**: User messages are now added to history before the API request, ensuring they are recorded even if the request fails.
- **Fixed empty result handling in continuation**: Empty `ChatResult` objects are now properly filtered out during continuation merging.
- **Fixed docstring formatting**: Resolved reStructuredText formatting issues in docstrings that caused documentation build warnings.

### Removed

- **Removed `auto_history` parameter** from `Chat.__init__()`
- **Removed `Chat.get_history()` method**
- **Removed `Chat.clear_history()` method**
- **Removed `Chat.clear_last_assistant_message()` method**
- **Removed obsolete test files**: `test_chat_auto_history.py`, `test_chat_new_features.py` (replaced with v2.0 tests)
- **Removed obsolete documentation**: `auto_history.rst` and related examples

### Documentation

- **Comprehensive documentation updates**: All documentation has been updated to reflect the new explicit history management approach
- **Migration guide**: Detailed examples showing how to migrate from v0.5.x to v2.0.0
- **New examples**: Updated all examples to use the new explicit history API
- **API reference**: Updated API reference documentation for all changed methods

### Testing

- **New comprehensive test suite**: Created new test files for v2.0.0 API:
  - `test_chat_v2.py` - Tests for `Chat` client's v2.0.0 API
  - `test_chat_history_v2.py` - Tests for `ChatHistory`'s `MutableSequence` protocol and new methods
  - `test_chat_continue_v2.py` - Tests for `ChatContinue`'s v2.0.0 API
  - `test_chat_streaming_continue_v2.py` - Tests for streaming continue edge cases
  - `test_chat_integration_v2.py` - Integration tests for v2.0.0 API
- **All tests passing**: Comprehensive test coverage ensuring correctness of the new architecture

## [0.5.1] - 2026-01-06

### Changed
- **BREAKING**: Simplified `Tokenizer` mode parameter from `mode="online"/"offline"/"auto_offline"` to `offline=True/False`
  - Removed `auto_offline` mode (which tried local cache first, then downloaded if not found)
  - Now uses `offline=True` for offline-only mode (fails if model not cached)
  - Now uses `offline=False` (default) for online mode (prioritizes local cache, downloads if needed)
  - Model download logic moved to business code, independent of `AutoTokenizer`
- Renamed exception classes to follow Python naming conventions:
  - `ChatStreamInterrupted` → `ChatStreamInterruptedError`
  - `ChatIncompleteResponse` → `ChatIncompleteResponseError`

### Added
- Added `huggingface-hub>=0.16.0` to `tokenizer` optional dependencies for explicit model downloading support
- `Tokenizer` now automatically downloads models when `offline=False` and model is not cached locally

### Fixed
- Fixed ruff linting configuration warnings by moving `select` and `ignore` to `[tool.ruff.lint]` section

## [0.5.0] - 2026-01-05

### Added
- Added `ChatHistory` class for comprehensive conversation history management
  - Automatic extraction from messages or Chat results (no manual maintenance required)
  - Support for `ChatHistory.from_messages()` and `ChatHistory.from_chat_result()` class methods
  - Token counting and analysis with `analyze_tokens()`, `count_tokens()`, and `count_tokens_per_round()`
  - Truncation by rounds with `truncate_by_rounds()` to fit context windows
  - Serialization to/from JSON with `to_json()` and `from_json()` methods
  - Round-based operations: `get_last_n_rounds()`, `remove_last_round()`, `update_last_assistant()`
  - Multi-format export support (Markdown, HTML, Text, JSON)
- Added `auto_history` feature to `Chat` class for automatic conversation tracking
  - Enable with `Chat(..., auto_history=True)` for zero-maintenance history recording
  - Automatically records all conversations (both streaming and non-streaming)
  - Access recorded history with `chat.get_history()` method
  - Clear history with `chat.clear_history()` method
  - Works seamlessly with streaming responses, updating history in real-time
- Added `ChatContinue` class for continuing cut-off responses
  - `ChatContinue.continue_request()` method to continue generation when `finish_reason == "length"`
  - Support for adding continue prompts (`add_continue_prompt=True`) or direct continuation (`add_continue_prompt=False`)
  - Customizable continue prompt text via `continue_prompt` parameter
  - `ChatContinue.merge_results()` method to merge multiple results into a single complete response
  - Automatically merges text, usage statistics, and metadata from multiple continuation requests

## [0.4.0] - 2026-01-05

### Added
- Added `finish_reason` field to `ChatResult` and `ChatStreamChunk` to track why generation stopped
  - Possible values: `"stop"`, `"length"`, `"content_filter"`, or `None`
  - Helps distinguish between normal completion, token limit, and content filtering
- Added `proxies` parameter to `Chat`, `Embed`, and `Rerank` classes for explicit proxy configuration
  - Supports environment variables (default behavior)
  - Allows explicit proxy configuration via `proxies` parameter
  - Can disable proxies by passing empty dict `{}`
- Added comprehensive integration tests for `finish_reason` functionality
- Added defensive handling for invalid `finish_reason` values from compatible services

### Improved
- Enhanced robustness of `finish_reason` parsing with normalization function
  - Handles empty strings, invalid types, and missing values gracefully
  - Ensures compatibility with services that don't fully implement OpenAI standard
- Improved error handling for malformed API responses
- Updated test configuration to use new endpoint keys (`embedding`, `reranker`, `completion`)

### Fixed
- Fixed test configuration key names to match updated `test_endpoints.json` structure
- Fixed potential issues with proxy configuration not being passed to requests

## [0.3.1] - 2026-01-04

### Changed
- Fixed CI workflow
- Update black tool python version dependencies

## [0.3.0] - 2026-01-03

### Changed
- Updated Python version support to 3.8-3.14
- Integrated CI workflow with uv for automated testing and building

## [0.2.0] - 2026-01-03

### Changed
- **BREAKING**: Removed chat-based rerank mode support. Rerank now only supports OpenAI-compatible and DashScope modes.
- Changed default rerank mode from `"chat"` to `"openai"`.
- Reorganized tests: all real API tests are now marked as `@pytest.mark.integration` and excluded from default test runs.

### Removed
- `ChatBasedHandler` class and chat-based rerank mode (`mode="chat"`).
- `chat_rerank_spec.rst` documentation (no longer needed as chat mode is removed).

### Improved
- Updated test endpoints to use `rerank_local_qwen3` and `embed_local_qwen3` for integration tests.
- Improved test organization following varlord's pattern for integration tests.
- Updated documentation to reflect rerank mode changes.

## [0.1.2] - 2025-12-29

### Added
- Scripts to automatically generate release notes
- Better github action workflow

## [0.1.1] - 2025-12-29

### Added
- Examples
- Comprehensive test suite
- Documentation

## [0.1.0] - 2025-12-28

### Added
- Initial release
- Chat API support with streaming
- Embedding API support
- Rerank API support
- Tokenizer support (optional dependency on transformers)
- Unified Usage and ResultBase classes
- Documentation

