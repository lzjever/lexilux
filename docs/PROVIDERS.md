# Multi-Provider Support

**Lexilux** is designed with one core philosophy: **One client, multiple providers**.

## Philosophy

The LLM ecosystem is rapidly evolving, with new providers emerging constantly. Rather than maintaining separate SDKs for each provider, Lexilux provides a unified client that works across all OpenAI-compatible APIs.

### Design Principles

1. **OpenAI-Compatible First**: We prioritize support for providers that implement OpenAI-compatible APIs
2. **Minimal Configuration**: Switching providers should only require changing `base_url` and `api_key`
3. **Consistent Interface**: Same API surface regardless of the underlying provider
4. **Best Effort Compatibility**: Graceful fallback for providers with slight API variations

## Supported Providers

### Tier 1: Fully Supported (OpenAI-Compatible)

These providers implement the OpenAI API specification and work seamlessly with Lexilux:

| Provider | Base URL | Notes |
|----------|----------|-------|
| **OpenAI** | `https://api.openai.com/v1` | Native support, reference implementation |
| **DeepSeek** | `https://api.deepseek.com` | Full OpenAI compatibility |
| **GLM / ZhipuAI** | `https://open.bigmodel.cn/api/paas/v4` | GLM-4.5/4.6 series |
| **Kimi / Moonshot** | `https://api.moonshot.cn/v1` | Moonshot AI |
| **Minimax** | `https://api.minimax.chat/v1` | Minimax models |
| **Qwen / Alibaba** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Tongyi Qianwen |
| **Azure OpenAI** | `https://YOUR_RESOURCE.openai.azure.com` | Microsoft Azure |
| **Groq** | `https://api.groq.com/openai/v1` | Ultra-fast inference |
| **Together AI** | `https://api.together.xyz/v1` | Open models hosting |
| **Fireworks AI** | `https://api.fireworks.ai/inference/v1` | Fast inference |
| **Anyscale** | `https://api.endpoints.anyscale.com/v1` | Ray-based serving |

### Tier 2: Adapter Required

These providers have different API formats but can be used with configuration:

| Provider | Notes |
|----------|-------|
| **Anthropic** | Different message format; use dedicated adapter or wait for native support |

## Quick Start Examples

### OpenAI

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o"
)
result = chat("Hello!")
```

### DeepSeek

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.deepseek.com",
    api_key="sk-...",
    model="deepseek-chat"
)
result = chat("Hello!")
```

### GLM / ZhipuAI

```python
from lexilux import Chat

chat = Chat(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="...",
    model="glm-4-plus"
)
result = chat("你好！")
```

### Kimi / Moonshot

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.moonshot.cn/v1",
    api_key="sk-...",
    model="moonshot-v1-8k"
)
result = chat("Hello!")
```

### Minimax

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.minimax.chat/v1",
    api_key="...",
    model="abab6.5-chat"
)
result = chat("Hello!")
```

### Qwen / Alibaba

```python
from lexilux import Chat

chat = Chat(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-...",
    model="qwen-turbo"
)
result = chat("你好！")
```

### Groq (Ultra-Fast Inference)

```python
from lexilux import Chat

chat = Chat(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
    model="llama-3.3-70b-versatile"
)
result = chat("Hello!")
```

## Environment Variables

For convenience, you can use environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# GLM / ZhipuAI
export ZHIPU_API_KEY="..."

# Kimi / Moonshot
export MOONSHOT_API_KEY="sk-..."

# Minimax
export MINIMAX_API_KEY="..."

# Qwen / Alibaba
export DASHSCOPE_API_KEY="sk-..."
```

Then in code:

```python
import os
from lexilux import Chat

chat = Chat(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat"
)
```

## Provider-Specific Considerations

### Rate Limits

Different providers have different rate limits. Configure accordingly:

```python
# Conservative for most providers
chat = Chat(..., rate_limit=5)  # 5 requests/second

# Aggressive for high-limit providers
chat = Chat(..., rate_limit=50)  # 50 requests/second
```

### Connection Pooling

Adjust pool size based on expected concurrency:

```python
# Low concurrency
chat = Chat(..., pool_size=2)

# High concurrency
chat = Chat(..., pool_size=10)
```

### Model Capabilities

Use the Model Registry to check capabilities:

```python
from lexilux import ModelRegistry

registry = ModelRegistry.get_instance()

# Check if model supports tool calling
spec = registry.get("deepseek-chat", provider="deepseek")
if spec.capabilities.tool_call:
    print("Supports function calling")
```

## Testing with Different Providers

When testing, you can easily switch between providers:

```python
import os

# Configuration map
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4o-mini"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "model": "deepseek-chat"
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.environ.get("ZHIPU_API_KEY"),
        "model": "glm-4-flash"
    },
}

# Select provider via environment variable
provider = os.environ.get("LLM_PROVIDER", "openai")
config = PROVIDERS[provider]

chat = Chat(**config)
```

## Contributing Provider Support

If you find a provider that works with Lexilux but isn't listed, please submit a PR to update this document. If you encounter compatibility issues, please open an issue with:

1. Provider name and base URL
2. Error message or unexpected behavior
3. API documentation link (if available)

## Future Roadmap

1. **Native Anthropic Support**: Direct API adapter
2. **Model Registry Expansion**: Comprehensive provider/model database
3. **Automatic Provider Detection**: Smart routing based on model name
4. **Cost Optimization**: Automatic selection of cheapest provider for a model
