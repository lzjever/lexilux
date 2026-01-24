# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-24
**Commit:** Current
**Branch:** main

## OVERVIEW
Test suite with 35 files covering all modules (Chat, Embed, Rerank, Tokenizer, History, Registry).

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Test fixtures | tests/conftest.py | test_config, has_real_api_config, proxy clearing |
| HTTP mocking | @responses.activate decorator | Mock external API calls |
| Async mocking | unittest.mock.patch.object | Mock async HTTP client methods |
| Streaming tests | tests/test_chat_streaming.py | StreamingIterator, StreamingResult patterns |
| Integration tests | tests/test_integration.py | Real API calls with @pytest.mark.integration |
| Function calling | tests/test_function_calling.py | OpenAI tool calling patterns |
| Async patterns | tests/test_async.py | @pytest.mark.asyncio, async def |

## CONVENTIONS

### Test Organization
```python
class TestChatCall:
    @pytest.fixture
    def chat(self):
        return Chat(base_url="...", api_key="...", model="gpt-4")

    @responses.activate  # HTTP mocking for sync tests
    def test_basic(self, chat):
        responses.add(responses.POST, "...", json={"...": "..."}, status=200)
        result = chat("Hello")

@pytest.mark.asyncio  # Required for async tests
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
@pytest.mark.skip_if_no_config  # Skip if test_endpoints.json missing
def test_real_api_call(test_config):
    """Requires real API credentials in test_endpoints.json"""
```

## ANTI-PATTERNS (THIS MODULE)

### Versioned Test Files
- **DO NOT** create multiple versions: `_v2`, `_clean`, `_fixed`, `_final` suffixes
- Keep only canonical versions: `test_async.py`, `test_chat.py`, `test_chat_streaming.py`
- Examples to clean up:
  - `test_async_clean_final.py`, `test_async_clean.py`, `test_async_fixed.py` → keep `test_async.py`
  - `test_chat_continue_v2.py` → merge to `test_chat_continue.py`
  - `test_chat_streaming_continue_v2.py` → merge to `test_chat_streaming.py`
  - `test_chat_v2.py`, `test_chat_history_v2.py` → merge to canonical versions

### Type Suppression
- test_chat.py uses `# type: ignore` for invalid input tests (acceptable)
- Only acceptable when testing error handling/invalid input scenarios
- Must include comment explaining why suppression is needed

### Missing Markers
- All integration tests MUST use `@pytest.mark.integration`
- Use `@pytest.mark.skip_if_no_config` for tests requiring API credentials
- Run with: `pytest -m integration` or `make test-integration`

### Proxy Configuration
- conftest.py automatically clears ALL proxy environment variables
- Do NOT set proxy environment variables in tests (will interfere)
- Direct connection ensured for all test HTTP requests
