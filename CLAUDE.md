# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lexilux is a unified LLM API client library for Python that provides function-like APIs for Chat, Embedding, Rerank, and Tokenizer services. It supports OpenAI-compatible endpoints with streaming, async, rate limiting, and connection pooling.

## Core Philosophy: One Client, Multiple Providers

Lexilux is designed to work seamlessly across all major LLM providers through their OpenAI-compatible APIs. The goal is to minimize provider lock-in by providing a unified interface.

**Tier 1 Providers (Fully Supported):**
- OpenAI, DeepSeek, GLM (ZhipuAI), Kimi (Moonshot), Minimax, Qwen (Alibaba), Groq, Azure OpenAI, Together AI, Fireworks AI

**Key Principle:** Switching providers should only require changing `base_url` and `api_key`, not rewriting application code.

See `docs/PROVIDERS.md` for complete provider documentation.

## Development Commands

### Setup
```bash
make dev-install          # Install package + all dev dependencies (recommended)
make setup-venv           # Install dependencies only (for CI/tools)
```

### Testing
```bash
make test                 # Unit tests in parallel (excludes integration)
make test-integration     # Integration tests (requires external services)
make test-cov             # Tests with HTML coverage report

# Run specific test file
uv run pytest tests/test_chat.py -v

# Run single test
uv run pytest tests/test_chat.py::TestChatCall::test_basic -v
```

### Code Quality
```bash
make lint                 # Ruff linting checks
make format               # Format code with ruff
make check                # Run all checks (lint + format + tests)
```

### Build & Docs
```bash
make build                # Build source and wheel distributions
make docs                 # Build Sphinx documentation
```

## Architecture

### Core Pattern: BaseAPIClient
All API clients inherit from `BaseAPIClient` (`lexilux/_base.py`):
- Sync HTTP via `requests`, Async via `httpx`
- Connection pooling, retry logic (tenacity), SSL control, timeout handling
- Context managers for resource cleanup

### Main Package Structure
```
lexilux/
├── chat/              # Primary module (~5,800 lines)
│   ├── client.py      # Main Chat class (sync/async __call__/stream)
│   ├── conversation.py # Conversation/ChatContinue (multi-turn)
│   ├── history.py     # ChatHistory (MutableSequence), TokenAnalysis
│   ├── _complete.py   # Auto-continue logic (finish_reason=="length")
│   ├── _request.py    # Request handling
│   ├── models.py      # ChatResult, ChatStreamChunk, ToolCall
│   ├── params.py      # ChatParams dataclass
│   └── streaming.py   # StreamingIterator
├── registry/          # Model registry, factory pattern
├── data/              # models.json (provider/model specifications)
├── _base.py           # BaseAPIClient (HTTP client foundation)
├── embed.py           # Embedding API
├── rerank.py          # Reranking API
├── tokenizer.py       # Tokenization (optional dependency)
└── exceptions.py      # Exception hierarchy
```

### Key Design Patterns

**Function-like API**: Clients are callable
```python
chat("hello")           # returns ChatResult
embed(["text"])         # returns EmbedResult
rerank("query", docs)   # returns RerankResult
```

**Sync/Async Pairs**: Every method has async variant
- `chat()` / `achat()`, `stream()` / `astream()`

**Message Normalization**: Accepts string, list, dict, or ChatHistory
- Internally normalized to `list[dict]` via `normalize_messages()`

**Result/Usage Tracking**: All results inherit from `ResultBase`
- Consistent `usage` attribute (input_tokens, output_tokens, total_tokens)

### Chat API Methods
| Method | Streaming | Ensures Complete | History Behavior |
|--------|-----------|------------------|------------------|
| `chat()` | No | No | Read-only |
| `stream()` | Yes | No | Read-only |
| `complete()` | No | Yes | Internal working copy |
| `complete_stream()` | Yes | Yes | Internal working copy |

## Code Conventions

- **Line length**: 100 characters
- **Type hints**: Required for all functions
- **Docstrings**: Google-style
- **Union types**: PEP 604 (`A | B`, not `Union[A, B]`)
- **Import order**: stdlib -> third-party -> local

## Testing Conventions

```python
# HTTP mocking (sync)
@responses.activate
def test_basic():
    responses.add(responses.POST, "...", json={...}, status=200)

# Async mocking
@pytest.mark.asyncio
async def test_async():
    with patch.object(chat, "_get_async_client") as mock:
        mock.return_value = AsyncMock()

# Integration tests (real API)
@pytest.mark.integration
@pytest.mark.skip_if_no_config
def test_real_api(test_config):
    ...
```

## Knowledge Base Files

- `lexilux/chat/AGENTS.md` - Chat module architecture and patterns
- `tests/AGENTS.md` - Test organization and conventions
