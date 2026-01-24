# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-24
**Commit:** 9600b79
**Branch:** main

## OVERVIEW
Unified LLM API client (v2.3.0) - Chat, Embedding, Rerank, Tokenizer with function-like API. Python 3.9+, uv for deps, pytest/ruff.

## STRUCTURE
```
lexilux/              # Main package (28 files, ~10,000 lines)
├── chat/              # 16 files, ~5,800 lines - PRIMARY HOTSPOT
│   ├── client.py       # 1174 lines - Main Chat (sync/async)
│   ├── conversation.py # 984 lines - Conversation (was continue_.py)
│   ├── history.py      # 1039 lines - ChatHistory, TokenAnalysis
│   ├── _complete.py    # 278 lines - Auto-continue (extracted)
│   ├── _request.py     # 238 lines - Request handling (extracted)
│   ├── formatters.py   # 381 lines - ChatHistoryFormatter
│   ├── models.py       # 322 lines - ChatResult, ChatStreamChunk, ToolCall
│   ├── params.py       # 216 lines - ChatParams dataclass
│   ├── tool_helpers.py # 239 lines - Tool helpers
│   ├── utils.py        # 181 lines - normalize_messages, parse_usage
│   ├── streaming.py    # 170 lines - StreamingIterator
│   ├── tools.py        # 142 lines - Tool, FunctionTool, ToolChoice
│   ├── content_blocks.py # 125 lines - Content blocks
│   └── exceptions.py   # 142 lines - Chat exceptions
├── registry/           # 4 files
│   ├── registry.py      # 18KB - ModelRegistry
│   ├── factory.py       # 20KB - ChatFactory, ConfiguredChat
│   └── models.py       # 6KB - ModelSpec, ProviderSpec
├── data/
│   └── models.json      # 920KB - Registry database
├── _base.py            # 22KB - BaseAPIClient (pooling, retry)
├── exceptions.py        # 6KB - Exception hierarchy
├── embed.py            # 14KB - Embedding
├── embed_params.py     # 3.3KB - EmbedParams
├── rerank.py          # 22KB - Rerank (OpenAI + DashScope)
├── tokenizer.py        # 17KB - Tokenizer (optional transformers)
└── usage.py           # 3.7KB - Usage tracking

tests/                 # 28 files (~9,300 lines)
docs/                  # Sphinx
examples/               # 16 demos
scripts/                # Utilities
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main Chat | chat/client.py | 1174 lines, sync/async __call__/stream |
| Chat history | chat/history.py | 1039 lines, MutableSequence, TokenAnalysis |
| Conversation | chat/conversation.py | 984 lines (was continue_.py) |
| Auto-continue | chat/_complete.py | 278 lines, finish_reason=="length" |
| Request handling | chat/_request.py | 238 lines (extracted) |
| Registry | registry/registry.py | ModelRegistry singleton |
| Base client | _base.py | Pooling, retry |
| Exceptions | exceptions.py | message, code, retryable |
| Tests | tests/ | @responses.activate for mocks |
| Build | Makefile | Uses uv if available |

## CONVENTIONS

### Import Order (ruff I001)
```python
from __future__ import annotations

from typing import Any
from collections.abc import Iterator, Sequence
import requests
from lexilux.exceptions import APIError
if TYPE_CHECKING:
    pass
```
No blank lines between groups.

### Type Hints (PEP 604)
- `str | None` NOT `Union[str, None]`
- `list[str]` NOT `List[str]`
- Keyword-only args with `*` preferred

### Naming
- Classes: PascalCase, Functions: snake_case, Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore
- Private modules: _leading_underscore.py (e.g., _complete.py, _request.py)

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
All exceptions have: message, code, retryable.

### Testing
- `@responses.activate` for HTTP mocks
- Class-based organization
- Integration: `@pytest.mark.integration`
- Run: `pytest -m "not integration"` (excludes by default)

### Dependencies
- Uses `uv` (Rust-based, fast)
- Lockfile `uv.lock` committed
- Dependency groups: `dev`, `docs`
- Optional extras: `tokenizer` (transformers, tokenizers, huggingface-hub)

## ANTI-PATTERNS

### Security (CRITICAL)
- **NEVER** commit API keys or credentials
- Blocked: `test_endpoints.json`, `endpoints.txt`, `*.secrets`
- Rotate exposed keys immediately
- **URGENT**: `endpoints.txt` has exposed key - ROTATE IMMEDIATELY

### Code Duplication
- **NEVER** add new sync/async pairs without consolidating shared logic
- **ALWAYS** consolidate parse_usage (4 duplicates exist)
- Legacy `lexilux/chat.py` and `lexilux/chat_params.py` have been REMOVED

### Type Suppression
- **NO type suppression without justification**
- 2 instances: registry/registry.py:254, tests/test_chat.py:98
- Only acceptable for error handling/invalid input tests

### Old-Style Union
- **USE PEP 604**: `A | B` NOT `Union[A, B]`
- 5+ instances in chat/ - migrate to `A | B`

### Bare Except
- **AVOID bare `except Exception:`** - 15 instances
- Locations: chat/conversation.py (6), chat/_complete.py (4), tokenizer.py (2), chat/tool_helpers.py (2), registry.py (1)
- Add logging or specific exceptions

## UNIQUE STYLES

1. Function-like API: `chat("hi")`, `embed(["text"])`, `rerank("query", docs)`
2. History immutability: Never modified internally (clone created)
3. Dual HTTP: requests (sync) + httpx (async)
4. OpenAI-compatible: Follows OpenAI API format
5. Connection pooling: 10 connections, 10 maxsize (default)
6. Retry logic: Exponential backoff, configurable max_retries
7. Coverage: 68% minimum (enforced in CI)
8. Backward compatibility: `ChatContinue` alias for `Conversation`
9. Modular chat: Subpackage with own __init__.py
10. Private helpers: `_complete.py`, `_request.py` (leading underscore)

## COMMANDS
```bash
make dev-install    # Install package + deps (uv)
make setup-venv     # Deps only (CI/CD)

make test            # Unit tests (excludes integration)
make test-integration  # Integration tests
make test-cov        # With coverage

make lint            # ruff check
make format          # ruff format
make check           # lint + format-check + test

make pre-commit-install   # Install hooks
make pre-commit-run      # Run manually
```

## NOTES

### Recent Refactoring (9600b79)
- **continue_.py → conversation.py**: Renamed
- **Helper extraction**: `_complete.py` (278 lines), `_request.py` (238 lines) extracted
- **Legacy removed**: `lexilux/chat.py`, `lexilux/chat_params.py`
- **Client reduced**: ~1900 → 1174 lines

### Known Issues
- `endpoints.txt` exposed key - **ROTATE IMMEDIATELY**
- 15 bare `except Exception:` clauses
- 5+ old-style Union[] types
- 4 parse_usage duplicates

### Test Cleanup
- Consolidate streaming: `test_chat_streaming.py` + `test_chat_stream.py`
- Merge history: `test_chat_history.py` + `test_chat_history_token_analysis.py`
- Remove versioned: `_v2`, `_clean`, `_fixed`, `_final` suffixes

### Root Clutter
- Delete: `bandit-report.json`, `test_output_full.log`, `.coverage`, `htmlcov/`, `lexilux.egg-info/`
- Move: `models_dev_api.json`, `models_dev_api_report.md` to `scripts/.dev/`
- Delete: `endpoints.txt` (exposed key!)

### Coverage
- Minimum: 68% overall (--cov-fail-under=68)
- Reports: HTML, terminal, term-missing, XML
- Pytest-xdist: `-n auto` (parallel)
