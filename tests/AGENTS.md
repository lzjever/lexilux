# TEST MODULE KNOWLEDGE BASE

**Generated:** 2026-01-24
**Commit:** 9600b79
**Branch:** main

## OVERVIEW
28 test files (~9,300 lines) - Chat, Embed, Rerank, Tokenizer, History, Registry.

## STRUCTURE
```
tests/
├── conftest.py                    # Fixtures, proxy clearing, markers
├── test_chat.py                   # Sync chat (@responses.activate)
├── test_async.py                 # Async API (AsyncMock, patch)
├── test_chat_stream.py           # Streaming (Mock, patch)
├── test_chat_streaming.py         # Streaming patterns
├── test_chat_history.py          # History management
├── test_chat_history_token_analysis.py # Token counting
├── test_chat_continue.py          # Conversation/continue tests
├── test_chat_api_improvements.py  # API enhancements
├── test_chat_params_additional.py  # Parameter tests
├── test_chat_params_integration.py # Real API param tests
├── test_chat_exceptions.py        # Error handling
├── test_chat_formatters.py        # Format utilities
├── test_embed.py                  # Embedding (@responses.activate)
├── test_embed_params.py           # Embed parameters
├── test_rerank.py                # Reranking (@responses.activate)
├── test_rerank_modes_consistency.py # Mode consistency
├── test_rerank_all_modes.py      # All mode tests
├── test_tokenizer.py             # Tokenizer (MagicMock, patch)
├── test_registry.py              # Model registry (fixtures, caplog, monkeypatch)
├── test_function_calling.py       # Unit tests for tools
├── test_function_calling_integration.py # Real API tool tests
├── test_integration.py            # Basic integration flows
├── test_all_features.py          # Comprehensive feature tests
├── test_exceptions.py            # Exception hierarchy
├── test_usage.py                # Usage tracking
├── test_chathistory_deepcopy.py  # Immutability tests
├── test_zhipu_multimodal.py      # Multimodal tests
└── verify_improvements.py        # Verification script
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Fixtures | conftest.py | test_config, has_real_api_config, proxy clearing |
| HTTP mocking | @responses.activate | Mock external API calls |
| Async mocking | unittest.mock.patch.object | Mock async HTTP client |
| Streaming | test_chat_streaming.py | StreamingIterator, StreamingResult |
| Integration | test_integration.py | Real API calls (@pytest.mark.integration) |
| Function calling | test_function_calling.py | OpenAI tool calling |
| Async | test_async.py | @pytest.mark.asyncio, async def |

## CONVENTIONS

### Test Organization
```python
class TestChatCall:
    @pytest.fixture
    def chat(self):
        return Chat(base_url="...", api_key="...", model="gpt-4")

    @responses.activate  # HTTP mocking (sync)
    def test_basic(self, chat):
        responses.add(responses.POST, "...", json={"...": "..."}, status=200)
        result = chat("Hello")

@pytest.mark.asyncio  # Required for async
async def test_acall_basic():
    with patch.object(chat, "_get_async_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client
        result = await chat.acall("Hello")
```

### Integration Tests
```python
@pytest.mark.integration
@pytest.mark.skip_if_no_config  # Skip if no test_endpoints.json
def test_real_api_call(test_config):
    """Requires real API credentials"""
```

### Proxy Config
- conftest.py clears ALL proxy env vars
- Do NOT set proxy vars in tests
- Direct connection ensured

### Execution
```bash
pytest tests/                   # Unit tests (default)
pytest tests/ -m integration     # Integration tests
pytest tests/ -n auto           # Parallel (default in Makefile)
pytest tests/ --cov=lexilux --cov-report=html  # Coverage
```

## ANTI-PATTERNS

### Versioned Test Files
- **DO NOT** create: `_v2`, `_clean`, `_fixed`, `_final` suffixes
- Keep canonical: `test_async.py`, `test_chat.py`, `test_chat_streaming.py`
- Clean up:
  - `test_async_clean_final.py`, `test_async_clean.py`, `test_async_fixed.py` → keep `test_async.py`
  - `test_chat_continue_v2.py` → merge to `test_chat_continue.py`
  - `test_chat_streaming_continue_v2.py` → merge to `test_chat_streaming.py`
  - `test_chat_v2.py`, `test_chat_history_v2.py` → merge to canonical

### Type Suppression
- test_chat.py uses `# type: ignore` for invalid input tests (acceptable)
- Only for error handling/invalid input scenarios
- Must include comment explaining why

### Missing Markers
- Integration tests MUST use `@pytest.mark.integration`
- Use `@pytest.mark.skip_if_no_config` for API credentials
- Run: `pytest -m integration`

### Utility Scripts in tests/
- `test_all_features.py` - Not a unit test
- `verify_improvements.py` - Verification script
- Move to `scripts/` or `tests/scripts/`

### Duplicate Tests
- `test_chat_streaming.py` (293 lines) + `test_chat_stream.py` (306 lines) - Both test streaming
- `test_chat_history.py` + `test_chat_history_token_analysis.py` - Same module
- Consolidate to canonical versions

## PYTEST CONFIG

### Markers (pytest.ini)
- `unit`: 单元测试
- `integration`: 集成测试
- `slow`: 慢速测试
- `mock`: 需要 mock 的测试
- `asyncio`: 异步测试

### Coverage
- Minimum: 68% (--cov-fail-under=68)
- Reports: HTML, terminal, term-missing, XML
- Codecov upload: Python 3.14 only

### Parallel
- Default: `-n auto` (pytest-xdist in Makefile)

## TEST INFRASTRUCTURE

### Shared Fixtures (conftest.py)
- `test_config()` - Loads test_endpoints.json (API credentials)
- `has_real_api_config` - Boolean flag for API config

### Mocking
- `responses` - HTTP mocking (43 uses across 5 files)
- `unittest.mock.AsyncMock` - Async method mocking
- `unittest.mock.patch` - Context manager patching
- `unittest.mock.MagicMock` - General mocking

### Stats
- 28 test files, ~9,300 lines
- 46 `@responses.activate` decorators (5 files)
- 135+ `@pytest.mark.asyncio` markers (mostly test_async.py)
- 14 `@pytest.fixture` definitions
- 99+ `pytest.raises` calls
- 1,192+ assertions

## CI/CD

### GitHub Actions (ci.yml)
- Tests: Python 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
- Uses uv for deps
- Command: `pytest tests/ --cov=lexilux --cov-report=xml --cov-report=term-missing --cov-report=html -v`

### Pre-commit Hooks
- ruff check
- ruff format --check
- trailing-whitespace-fixer
- end-of-file-fixer
- check-yaml

## RECENT CHANGES (9600b79)

### Chat Module Refactoring
- **conversation.py rename**: Tests updated (continue_.py → conversation.py)
- **New helper modules**: Tests may need updates for _complete.py, _request.py
- **Legacy removed**: Tests using lexilux/chat.py or lexilux/chat_params.py must migrate

### Test Cleanup Recommended
1. Remove versioned test files (_v2, _clean, _fixed, _final)
2. Consolidate streaming tests (test_chat_streaming.py + test_chat_stream.py)
3. Consolidate history tests (test_chat_history.py + test_chat_history_token_analysis.py)
4. Move utility scripts (test_all_features.py, verify_improvements.py) to scripts/
