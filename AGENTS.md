# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-24
**Commit:** Current
**Branch:** main

## OVERVIEW
Lexilux is a unified LLM API client library (v2.3.0) providing Chat, Embedding, Rerank, and Tokenizer support with a function-like API design. Uses Python 3.9+, uv for dependency management, pytest for testing, and ruff for linting.

## STRUCTURE
```
./
├── lexilux/          # Main package (24 files, ~9,657 lines)
│   ├── chat/          # Chat submodule (12 files) - PRIMARY COMPLEXITY HOTSPOT
│   ├── registry/       # Model registry (4 files)
│   ├── _base.py        # BaseAPIClient with connection pooling
│   ├── exceptions.py    # Unified exception hierarchy
│   ├── usage.py        # Usage tracking
│   ├── chat.py         # LEGACY Chat (delegates to chat/)
│   ├── chat_params.py  # LEGACY params (duplicate)
│   ├── embed.py        # Embedding client
│   ├── rerank.py       # Rerank client (OpenAI + DashScope)
│   └── tokenizer.py    # Tokenizer client (optional transformers)
├── tests/             # 35 test files
├── docs/              # Sphinx documentation
├── examples/           # 16 demo scripts
├── scripts/           # Utility scripts
└── pyproject.toml     # Modern build config
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main Chat client | lexilux/chat/client.py | 1921 lines, complex, has sync/async duplication |
| Chat history | lexilux/chat/history.py | 1039 lines, MutableSequence, token analysis |
| Model registry | lexilux/registry/registry.py | Model lookup, singleton pattern |
| Base HTTP client | lexilux/_base.py | Connection pooling, retry logic |
| Exception handling | lexilux/exceptions.py | All errors have message, code, retryable |
| Testing patterns | tests/ | Use @responses.activate for HTTP mocks |
| Build commands | Makefile | Uses uv if available |

## CONVENTIONS

### Import Order (Enforced by ruff I001)
```python
from __future__ import annotations

from typing import Any
from collections.abc import Iterator, Sequence
import requests
from lexilux.exceptions import APIError
if TYPE_CHECKING:
    pass
```
**No blank lines between groups.**

### Type Hints (PEP 604)
- Use `str | None` NOT `Union[str, None]`
- Use `list[str]` NOT `List[str]`
- Keyword-only args with `*` preferred

### Naming
- Classes: PascalCase
- Functions: snake_case
- Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore

### Error Handling
```python
from lexilux.exceptions import LexiluxError, AuthenticationError, RateLimitError

try:
    result = chat("Hello")
except AuthenticationError as e:
    logger.error(f"Auth failed: {e.message}")
except RateLimitError as e:
    logger.warning(f"Rate limited: {e.message}")  # retryable=True
except LexiluxError as e:
    logger.error(f"Error {e.code}: {e.message}")
```
**All exceptions have: message, code, retryable attributes.**

### Testing
- Use `@responses.activate` for HTTP mocks
- Class-based test organization
- Descriptive docstrings
- Integration tests: `@pytest.mark.integration`
- Run with: `pytest -m "not integration"` (excludes by default)

## ANTI-PATTERNS (THIS PROJECT)

### Security
- **NEVER** commit API keys or test credentials
- Files blocked: `test_endpoints.json`, `endpoints.txt`, `*.secrets`
- Rotate exposed keys immediately if found

### Code Duplication
- **NEVER** add new sync/async method pairs without consolidating shared logic
- **ALWAYS** consolidate parse_usage implementations (4 duplicates exist)
- **DO NOT** use legacy `lexilux/chat.py` or `lexilux/chat_params.py`

### Type Suppression
- **NO type suppression without justification**
- Only 5 instances exist, all with reasons

### Bare Except
- **AVOID bare `except Exception:`** - 4 instances in `chat/continue_.py`
- Add logging or use specific exceptions

### Legacy Code
- `lexilux/chat.py` - DO NOT use (legacy, delegates to chat/)
- `lexilux/chat_params.py` - DO NOT use (duplicate of chat/params.py)
- Use modular chat submodule instead

## UNIQUE STYLES

1. **Function-like API**: `chat("hi")`, `embed(["text"])`, `rerank("query", docs)`
2. **History immutability**: When history passed to Chat methods, it's never modified (clone created internally)
3. **Dual HTTP clients**: requests (sync) + httpx (async) for full support
4. **OpenAI-compatible**: Follows OpenAI API format for messages, errors, usage
5. **Connection pooling**: Enabled by default (10 connections, 10 maxsize)
6. **Retry logic**: Exponential backoff with configurable max_retries
7. **Coverage requirements**: 60% minimum overall, 80% for core, 70% for utilities

## COMMANDS
```bash
# Setup
make dev-install    # Install package + deps (uv recommended)
make setup-venv     # Deps only (for CI/CD)

# Testing
make test            # Unit tests (excludes integration)
make test-integration  # Integration tests
make test-cov        # With coverage

# Quality
make lint            # ruff check
make format          # ruff format
make check           # lint + format-check + test

# Pre-commit
make pre-commit-install   # Install hooks
make pre-commit-run      # Run manually
```

## NOTES

### Known Issues
- `endpoints.txt` contains exposed API key - **ROTATE IMMEDIATELY**
- `chat/client.py` has sync/async duplication - 1921 lines, primary complexity hotspot
- 4 parse_usage implementations scattered across modules - consolidate when possible
- Legacy modules `lexilux/chat.py` and `lexilux/chat_params.py` - remove after migration verification

### Test Cleanup Needed
- Multiple versioned test files: `test_async_clean_final.py`, `test_async_fixed.py`, `test_chat_continue_v2.py`, etc.
- Keep only canonical versions, archive or delete others

### Root Clutter
- Move `test_all_features.py` to `tests/`
- Move `models_dev_api.json`, `models_dev_api_report.md` to `scripts/` or `.dev/`
- Delete `test_output_full.log`, `bandit-report.json`

### Module Naming
- `chat/continue_.py` uses trailing underscore (Python keyword conflict)
- Consider renaming to `conversation.py` or `chatsession.py` if breaking changes acceptable

### Dependencies
- Uses uv for dependency management (Rust-based, fast)
- Lockfile `uv.lock` is committed for reproducible builds
- transformers, huggingface_hub optional (only for tokenizer)
