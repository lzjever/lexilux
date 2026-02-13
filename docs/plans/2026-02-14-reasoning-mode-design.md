# Reasoning Mode Support Design

**Date:** 2026-02-14
**Status:** Approved
**Target Version:** v2.8.0

## Overview

Add unified reasoning mode support to Lexilux, allowing users to enable extended thinking/reasoning across multiple LLM providers with a single, consistent API.

## Goals

1. **Unified API**: Single `reasoning=True` parameter works across all providers
2. **Data-driven**: Leverage models.json for model capabilities, code for provider config
3. **Minimal code**: ~200 lines total, no merge complexity
4. **Easy sync**: `make sync-models` just downloads, no transformation

## Non-Goals

- Backward compatibility (project not yet released)
- Provider-specific reasoning features beyond the unified abstraction
- Runtime reasoning configuration updates

## Architecture

```
lexilux/
├── data/
│   └── models.json              # Pure upstream data (synced from models.dev)
│
├── providers/                   # NEW: Provider-specific configs
│   ├── __init__.py
│   ├── base.py                  # ReasoningConfig dataclass
│   ├── registry.py              # PROVIDERS dict, get_reasoning_config()
│   ├── openai.py
│   ├── anthropic.py
│   ├── deepseek.py
│   ├── moonshotai.py
│   ├── zhipu.py
│   └── ...
│
├── chat/
│   ├── reasoning.py             # NEW: normalize_reasoning(), build_reasoning_request(), extract_reasoning_content()
│   ├── params.py                # MODIFIED: Add reasoning field
│   ├── models.py                # MODIFIED: Add reasoning to ChatResult, ChatStreamChunk
│   └── client.py                # MODIFIED: Integrate reasoning into request building
```

## Data Model

### ReasoningConfig (per provider)

```python
@dataclass(frozen=True)
class ReasoningConfig:
    method: str                    # "extra_body" | "reasoning_param" | "thinking_param" | "model_selection"
    response_field: str | None     # Field name in response
    default_effort: str = "medium"
    supports_budget: bool = False
    effort_to_budget: dict | None = None
    params: dict | None = None
```

### Provider Configurations

| Provider | Method | Response Field | Notes |
|----------|--------|----------------|-------|
| DeepSeek | `extra_body` | `reasoning_content` | `{"thinking": {"type": "enabled"}}` |
| OpenAI | `reasoning_param` | `null` (hidden) | `{"effort": "medium"}` |
| Anthropic | `thinking_param` | `thinking` | `{"type": "enabled", "budget_tokens": N}` |
| Kimi/Moonshot | `model_selection` | `reasoning_content` | Enabled by model name |
| GLM/Zhipu | `extra_body` | `reasoning_content` | `{"thinking": {"type": "enabled"}}` |

## User API

### Basic Usage

```python
from lexilux import Chat

chat = Chat(base_url="...", api_key="...", model="deepseek-reasoner")

# Simple on/off
result = chat("Solve this problem", reasoning=True)
print(result.reasoning)  # Reasoning content
print(result.text)       # Final answer

# With effort level
result = chat("Complex task", reasoning={"effort": "high"})

# With budget (Anthropic-style)
result = chat("Hard problem", reasoning={"effort": "high", "max_tokens": 16000})
```

### Streaming

```python
for chunk in chat.stream("Solve this", reasoning=True):
    if chunk.reasoning:
        print(chunk.reasoning, end="", flush=True)  # Reasoning delta
    if chunk.delta:
        print(chunk.delta, end="", flush=True)      # Answer delta
```

## Implementation Details

### ChatParams Extension

```python
@dataclass
class ChatParams:
    # ... existing fields ...
    reasoning: bool | dict | None = None
```

### ChatResult Extension

```python
@dataclass
class ChatResult:
    # ... existing fields ...
    reasoning: str | None = None
```

### ChatStreamChunk Extension

```python
@dataclass
class ChatStreamChunk:
    # ... existing fields ...
    reasoning: str = ""  # Delta for reasoning content
```

### Core Helper Functions

```python
# lexilux/chat/reasoning.py

def normalize_reasoning(reasoning: bool | dict | None) -> dict:
    """Convert reasoning param to normalized dict."""
    ...

def build_reasoning_request(provider_id: str, reasoning: dict) -> dict:
    """Build provider-specific request params."""
    ...

def extract_reasoning_content(response: dict, provider_id: str) -> str | None:
    """Extract reasoning text from response."""
    ...
```

## Provider Detection

Provider ID is determined from `base_url` or explicit `provider` parameter:

```python
PROVIDER_URL_PATTERNS = {
    "deepseek": ["deepseek.com"],
    "openai": ["openai.com"],
    "anthropic": ["anthropic.com", "claude.ai"],
    "moonshotai": ["moonshot.cn", "moonshot.ai"],
    "zhipu": ["bigmodel.cn", "zhipuai.cn"],
}
```

## Sync Mechanism

```makefile
# Makefile
sync-models: ## Sync models.json from models.dev
	curl -sL "https://models.dev/api.json" -o lexilux/data/models.json
```

No merge logic needed - models.json stays pure upstream data.

## File Changes Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `lexilux/chat/params.py` | Modify | +1 |
| `lexilux/chat/models.py` | Modify | +5 |
| `lexilux/chat/client.py` | Modify | +10 |
| `lexilux/chat/reasoning.py` | Create | ~40 |
| `lexilux/providers/__init__.py` | Create | ~10 |
| `lexilux/providers/base.py` | Create | ~15 |
| `lexilux/providers/registry.py` | Create | ~30 |
| `lexilux/providers/openai.py` | Create | ~10 |
| `lexilux/providers/anthropic.py` | Create | ~15 |
| `lexilux/providers/deepseek.py` | Create | ~10 |
| `lexilux/providers/moonshotai.py` | Create | ~10 |
| `lexilux/providers/zhipu.py` | Create | ~10 |
| `lexilux/__init__.py` | Modify | +1 (export) |
| `Makefile` | Modify | +3 |
| `tests/test_reasoning.py` | Create | ~100 |

**Total: ~270 lines**

## Testing Strategy

1. Unit tests for `normalize_reasoning()`, `build_reasoning_request()`, `extract_reasoning_content()`
2. Mock provider responses for each adapter
3. Integration tests with real APIs (marked as integration)

## Rollout Plan

1. Implement core reasoning.py and providers/ module
2. Modify ChatParams, ChatResult, ChatStreamChunk
3. Integrate into Chat client
4. Add tests
5. Update documentation
6. Release as v2.8.0

## Future Considerations

- Reasoning summary support (OpenAI Responses API)
- Encrypted reasoning blocks (Anthropic)
- Interleaved thinking with tools
- Per-model reasoning capability detection from models.json
