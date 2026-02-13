# Lexilux

[![PyPI version](https://img.shields.io/pypi/v/lexilux.svg)](https://pypi.org/project/lexilux/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/lexilux/badge/?version=latest)](https://lexilux.readthedocs.io)
[![CI](https://github.com/lzjever/lexilux/workflows/CI/badge.svg)](https://github.com/lzjever/lexilux/actions)
[![codecov](https://codecov.io/gh/lzjever/lexilux/branch/main/graph/badge.svg)](https://codecov.io/gh/lzjever/lexilux)

**Lexilux** is a unified LLM API client library that makes calling Chat, Embedding, Rerank, and Tokenizer APIs as simple as calling a function.

## Features

- **Function-like API**: Call APIs like functions (`chat("hi")`, `embed(["text"])`)
- **Streaming Support**: Built-in streaming for Chat with usage tracking
- **Unified Usage**: Consistent usage statistics across all APIs
- **Flexible Input**: Support multiple input formats (string, list, dict)
- **OpenAI-Compatible**: Works with OpenAI-compatible APIs
- **Automatic Retry**: Built-in retry logic with exponential backoff
- **Connection Pooling**: HTTP connection pooling for better performance
- **Rate Limiting**: Built-in rate limiter for API request throttling
- **SSL Control**: Configurable SSL certificate verification
- **Input Validation**: Comprehensive parameter validation with clear errors
- **Exception Hierarchy**: Comprehensive exception system with error codes
- **Function Calling**: OpenAI-compatible function/tool calling support
- **Multimodal Support**: Vision capabilities with image inputs
- **Async Support**: Full async/await API for concurrent operations

## One Client, Multiple Providers

Lexilux is designed to work seamlessly with all major LLM providers through their OpenAI-compatible APIs:

| Provider | Base URL |
|----------|----------|
| **OpenAI** | `https://api.openai.com/v1` |
| **DeepSeek** | `https://api.deepseek.com` |
| **GLM / ZhipuAI** | `https://open.bigmodel.cn/api/paas/v4` |
| **Kimi / Moonshot** | `https://api.moonshot.cn/v1` |
| **Minimax** | `https://api.minimax.chat/v1` |
| **Qwen / Alibaba** | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| **Groq** | `https://api.groq.com/openai/v1` |

Simply change `base_url` and `api_key` to switch providers:

```python
# OpenAI
chat = Chat(base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o")

# DeepSeek
chat = Chat(base_url="https://api.deepseek.com", api_key="sk-...", model="deepseek-chat")

# GLM (智谱)
chat = Chat(base_url="https://open.bigmodel.cn/api/paas/v4", api_key="...", model="glm-4-plus")
```

See [PROVIDERS.md](docs/PROVIDERS.md) for complete provider documentation.

## Installation

### Quick Install

```bash
pip install lexilux
```

### With Tokenizer Support

```bash
pip install lexilux[tokenizer]
```

### Development Setup with uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# For active development
make dev-install

# Or manually with uv
uv sync --group docs --all-extras
```

### Legacy pip Setup

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Chat

```python
from lexilux import Chat

chat = Chat(base_url="https://api.example.com/v1", api_key="your-key", model="gpt-4")
result = chat("Hello, world!")
print(result.text)
print(result.usage.total_tokens)
```

### Streaming

```python
for chunk in chat.stream("Tell me a joke"):
    print(chunk.delta, end="", flush=True)
    if chunk.done:
        print(f"\nTokens: {chunk.usage.total_tokens}")
```

### Error Handling

```python
from lexilux import LexiluxError, AuthenticationError, RateLimitError

try:
    result = chat("Hello, world!")
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
except RateLimitError as e:
    if e.retryable:
        print(f"Rate limited: {e.message}")
except LexiluxError as e:
    print(f"Error: {e.code} - {e.message}")
```

### Function Calling

```python
from lexilux import Chat, FunctionTool

get_weather = FunctionTool(
    name="get_weather",
    description="Get weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
)

result = chat("What's the weather in Paris?", tools=[get_weather])
if result.has_tool_calls:
    for tool_call in result.tool_calls:
        print(f"Calling: {tool_call.name}")
        print(f"Arguments: {tool_call.get_arguments()}")
```

### Rate Limiting

```python
from lexilux import Chat

# Limit to 10 requests per second
chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    model="gpt-4",
    rate_limit=10  # requests per second
)
```

### SSL Verification

```python
from lexilux import Chat

# Disable SSL verification for testing (not recommended for production)
chat = Chat(
    base_url="https://api.example.com/v1",
    api_key="your-key",
    model="gpt-4",
    verify_ssl=False
)
```

### Async

```python
import asyncio
from lexilux import Chat

async def main():
    chat = Chat(base_url="...", api_key="...", model="gpt-4")
    result = await chat.a("Hello, async world!")
    print(result.text)

asyncio.run(main())
```

### Connection Pooling

By default, Lexilux uses a connection pool with size 2 to reuse HTTP connections
and improve performance. You can customize this based on your API provider's limits:

```python
chat = Chat(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    model="gpt-4",
    pool_size=10,  # Increase for higher concurrency
)
```

**Provider Limits:**
- OpenAI: Recommended <= 10
- Anthropic: Recommended <= 5
- Other providers: Check their documentation

### Automatic Retries

Lexilux automatically retries failed requests with exponential backoff when:
- Rate limit errors (HTTP 429)
- Server errors (HTTP 500, 502, 503, 504)
- Network timeouts or connection errors

Configure retry behavior:

```python
chat = Chat(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    max_retries=3,  # Retry up to 3 times on transient errors
)
```

**Note:** Only `retryable=True` errors trigger automatic retries.
Authentication and validation errors are never retried.

### Chat API Selection Guide

| Method | Streaming | Ensures Complete | History Behavior |
|--------|-----------|------------------|------------------|
| `chat()` | No | No | Read-only |
| `stream()` | Yes | No | Read-only |
| `complete()` | No | Yes | Internal working copy |
| `complete_stream()` | Yes | Yes | Internal working copy |

**History Behavior:**
- `chat()` and `stream()` never modify your history object
- `complete()` methods create an internal working copy for state management
- Your original `ChatHistory` is always preserved

## Documentation

Full documentation available at: [lexilux.readthedocs.io](https://lexilux.readthedocs.io)

## Examples

Check out the `examples/` directory for practical examples:

- `examples/01_hello_world.py` - Basic chat completion
- `examples/02_system_message.py` - Using system messages
- `examples/10_streaming.py` - Streaming chat
- `examples/11_conversation.py` - Multi-turn conversations
- `examples/12_chat_params.py` - Custom chat parameters
- `examples/20_embedding.py` - Text embedding
- `examples/21_rerank.py` - Document reranking
- `examples/22_tokenizer.py` - Tokenization
- `examples/30_function_calling.py` - Function calling
- `examples/31_multimodal.py` - Vision capabilities
- `examples/32_async.py` - Async operations
- `examples/40_chat_history.py` - History management
- `examples/41_auto_continue.py` - Continue cut-off responses
- `examples/42_error_handling.py` - Error handling patterns
- `examples/43_custom_formatting.py` - Custom response formatting

Run examples:

```bash
python examples/01_hello_world.py
```

## Testing

```bash
# Run unit tests
make test

# Run integration tests
make test-integration

# Run with coverage
make test-cov

# Run linting
make lint

# Format code
make format
```

Build documentation locally:

```bash
cd docs && make html
```

## About Agentsmith

**Lexilux** is part of the **Agentsmith** open-source ecosystem. Agentsmith is a ToB AI agent and algorithm development platform, currently deployed in multiple highway management companies, securities firms, and regulatory agencies in China. The Agentsmith team is gradually open-sourcing the platform by removing proprietary code and algorithm modules, as well as enterprise-specific customizations, while decoupling the system for modular use by the open-source community.

### Agentsmith Open-Source Projects

- **[Varlord](https://github.com/lzjever/varlord)** - Configuration management library
- **[Routilux](https://github.com/lzjever/routilux)** - Event-driven workflow orchestration
- **[Serilux](https://github.com/lzjever/serilux)** - Flexible serialization framework
- **[Lexilux](https://github.com/lzjever/lexilux)** - Unified LLM API client library

## License

Lexilux is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

## Links

- **PyPI**: [pypi.org/project/lexilux](https://pypi.org/project/lexilux)
- **Documentation**: [lexilux.readthedocs.io](https://lexilux.readthedocs.io)
- **GitHub**: [github.com/lzjever/lexilux](https://github.com/lzjever/lexilux)
