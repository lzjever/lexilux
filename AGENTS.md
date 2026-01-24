# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-24
**Commit:** ca17f0d
**Branch:** main

## OVERVIEW
Lexilux - Unified LLM API client library (Chat, Embedding, Rerank, Tokenizer) with streaming, function calling, and multimodal support.

## STRUCTURE
```
lexilux/
├── lexilux/            # Main package (9k lines)
│   ├── chat/           # Chat API (5.8k lines) - PRIMARY HOTSPOT
│   │   ├── client.py   # Main Chat (1,174 lines)
│   │   ├── history.py  # ChatHistory, TokenAnalysis (1,039 lines)
│   │   ├── conversation.py  # Conversation/continue (984 lines)
│   │   ├── _complete.py    # Auto-continue (278 lines)
│   │   ├── _request.py     # Request handling (238 lines)
│   │   ├── streaming.py    # StreamingIterator
│   │   ├── tools.py        # Function calling
│   │   └── formatters.py   # ChatHistoryFormatter
│   ├── embed.py        # Embedding API
│   ├── rerank.py       # Reranking API (693 lines)
│   ├── tokenizer.py    # HuggingFace integration
│   ├── registry/       # Model registry
│   └── exceptions.py   # Error hierarchy
├── tests/              # Test suite (28 files, 457 tests)
├── docs/               # Sphinx documentation
├── Makefile            # Build/test commands
└── pyproject.toml      # Python project config
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main Chat API | lexilux/chat/client.py | Sync/async __call__/stream, 1,174 lines |
| Chat history | lexilux/chat/history.py | MutableSequence, TokenAnalysis |
| Conversation/continue | lexilux/chat/conversation.py | Auto-continue logic |
| Streaming | lexilux/chat/streaming.py | StreamingIterator |
| Function calling | lexilux/chat/tools.py, tool_helpers.py | FunctionTool, execute_tool_calls |
| Embedding | lexilux/embed.py | Vector embeddings |
| Reranking | lexilux/rerank.py | Search result reranking |
| Tokenizer | lexilux/tokenizer.py | HuggingFace Transformers |
| Model registry | lexilux/registry/registry.py | Provider management |

## CONVENTIONS

### Development
- Package manager: **uv** (with pip fallback)
- Linting/formatting: **ruff** (line-length: 100)
- Testing: **pytest** with responses (HTTP mocking), pytest-asyncio
- Coverage: 68% minimum
- Documentation: Sphinx with RTD theme
- Version from `lexilux/__init__.py` (dynamic setuptools)

### Sync/Async Pairs
- Pattern: `<method>` (sync) and `a<method>` (async)
- Example: `chat()`/`achat()`, `stream()`/`astream()`
- Both client.py and conversation.py have full sync/async versions

### Message Normalization
- Accepts: string, list, dict, ChatHistory
- Normalized to list[dict] internally
- Use `normalize_messages()` utility

### Streaming
- `StreamingIterator` (sync), `AsyncStreamingIterator` (async)
- Accumulate with `chunk.delta`
- `chunk.done` flag = completion

### Tool Calling
- Define: `FunctionTool` class
- Pass: `ChatParams(tools=[...])`
- Check: `result.has_tool_calls`
- Execute: `execute_tool_calls()` helper

## ANTI-PATTERNS (THIS PROJECT)

### Exception Handling
- **AVOID** bare `except Exception:` (10 instances in conversation.py, _complete.py)
- Use specific exceptions with logging and re-raise

### Code Duplication
- **NEVER** add sync/async pairs without consolidating shared logic
- Extract to private methods or helpers

### ChatStreamChunk
- **usage parameter REQUIRED** (not optional like ChatResult)
- Always pass Usage() instance

### History Immutability
- ChatHistory never modified internally when passed
- Clone created to preserve original

### Old-Style Union
- **USE PEP 604**: `A | B` NOT `Union[A, B]`
- Migrate instances in content_blocks.py:23,40; models.py:29; tools.py:57

### Test Naming
- **DO NOT** create versioned test files (_v2, _clean, _fixed, _final)
- Keep canonical: `test_async.py`, `test_chat.py`, `test_chat_streaming.py`

## UNIQUE STYLES

### Chat Module Refactoring (commit 9600b79)
- **continue_.py → conversation.py**: Renamed
- **_complete.py extracted**: 278 lines (auto-continue logic)
- **_request.py extracted**: 238 lines (request handling)
- **client.py reduced**: ~1900 → 1,174 lines
- `ChatContinue` alias for `Conversation` class (backward compatibility)

### Token Analysis
`TokenAnalysis` class tracks per-role token counts for compaction decisions.

### Auto-Continue
Automatic continuation when `finish_reason=="length"` - handles truncated responses.

## COMMANDS
```bash
make dev-install          # Install with uv
make test                 # Run unit tests
make test-integration     # Run integration tests
make test-cov             # Coverage report
make lint                 # ruff check
make format               # ruff format
make check                # Run all checks
```

## NOTES

### Configuration
- Model registry tracks provider capabilities
- Tokenizer is optional dependency (requires transformers)
- Uses `local_files_only=True` for tokenizer to avoid network calls

### Testing
- Unit tests use `@responses.activate` for HTTP mocking
- Integration tests marked with `@pytest.mark.integration`
- Use `test_endpoints.json` for real API credentials
- Proxy env vars cleared in conftest.py

### Module Exports (from lexilux/__init__.py)
- **Classes**: Chat, ChatResult, ChatStreamChunk, ChatParams, ChatHistory, Conversation
- **Tools**: Tool, FunctionTool, ToolChoice, execute_tool_calls
- **Multimodal**: ContentBlock, TextContentBlock, ImageContentBlock
- **Utils**: normalize_messages, merge_histories
