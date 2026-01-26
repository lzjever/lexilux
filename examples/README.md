# Lexilux Examples

This directory contains practical examples demonstrating Lexilux usage, organized by difficulty level from beginner to expert.

## Quick Start

1. Configure your API endpoints (see [Configuration](#configuration) below)
2. Run any example:
   ```bash
   python examples/01_hello_world.py
   ```

## Examples by Level

### Level 1: Getting Started (Beginner)

**Start here if you're new to Lexilux!**

| File | Description | Concepts |
|------|-------------|----------|
| [`01_hello_world.py`](01_hello_world.py) | Simplest possible chat call | Basic chat completion, usage tracking |
| [`02_system_message.py`](02_system_message.py) | Control AI behavior with system messages | System prompts, personality control |

```bash
python examples/01_hello_world.py
python examples/02_system_message.py
```

---

### Level 2: Core Features (Intermediate)

**Essential features for most use cases.**

| File | Description | Concepts |
|------|-------------|----------|
| [`10_streaming.py`](10_streaming.py) | Real-time response streaming | Streaming, chunk processing, usage tracking |
| [`11_conversation.py`](11_conversation.py) | Multi-turn conversations | Message history, chat loop, context management |
| [`12_chat_params.py`](12_chat_params.py) | Control response behavior | Temperature, max_tokens, stop sequences, ChatParams |

```bash
python examples/10_streaming.py
python examples/11_conversation.py
python examples/12_chat_params.py
```

---

### Level 3: Other APIs (Intermediate)

**Features beyond chat completions.**

| File | Description | Concepts |
|------|-------------|----------|
| [`20_embedding.py`](20_embedding.py) | Text to vector conversion | Embeddings, semantic similarity, batch processing |
| [`21_rerank.py`](21_rerank.py) | Document reranking | Relevance sorting, top-k filtering, search |
| [`22_tokenizer.py`](22_tokenizer.py) | Token counting | Cost estimation, input validation |

```bash
python examples/20_embedding.py
python examples/21_rerank.py
python examples/22_tokenizer.py
```

**Note:** Tokenizer requires extra dependencies:
```bash
pip install lexilux[tokenizer]
```

---

### Level 4: Advanced Features (Advanced)

**Powerful capabilities for complex applications.**

| File | Description | Concepts |
|------|-------------|----------|
| [`30_function_calling.py`](30_function_calling.py) | Let AI call your functions | Function tools, parallel execution, tool loops |
| [`31_multimodal.py`](31_multimodal.py) | Process images with text | Vision, image URLs, base64 encoding, OCR |
| [`32_async.py`](32_async.py) | Concurrent API calls | Async/await, parallel requests, rate limiting |

```bash
python examples/30_function_calling.py
python examples/31_multimodal.py
python examples/32_async.py
```

---

### Level 5: Expert Topics (Expert)

**Advanced techniques for production applications.**

| File | Description | Concepts |
|------|-------------|----------|
| [`40_chat_history.py`](40_chat_history.py) | Advanced conversation management | ChatHistory, formatting, searching, merging |
| [`41_auto_continue.py`](41_auto_continue.py) | Handle truncated responses | Auto-continue, streaming continue, completion guarantees |
| [`42_error_handling.py`](42_error_handling.py) | Robust error management | Exception hierarchy, retry logic, input validation |
| [`43_custom_formatting.py`](43_custom_formatting.py) | Export conversations | Markdown, HTML, JSON, custom formatters |

```bash
python examples/40_chat_history.py
python examples/41_auto_continue.py
python examples/42_error_handling.py
python examples/43_custom_formatting.py
```

---

## Configuration

Most examples require API credentials. Configure them in one of two ways:

### Option 1: Configuration File (Recommended for Testing)

Create `tests/test_endpoints.json` or `examples/test_endpoints.json`:

```json
{
  "completion": {
    "model": "gpt-4",
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-your-api-key"
  },
  "embedding": {
    "model": "text-embedding-ada-002",
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-your-api-key"
  },
  "reranker": {
    "model": "rerank-model",
    "api_base": "https://api.example.com/v1",
    "api_key": "sk-your-api-key",
    "mode": "openai"
  }
}
```

### Option 2: Custom Config Path

```bash
python examples/01_hello_world.py --config /path/to/config.json
```

### Option 3: Environment Variables (Production)

For production, use environment variables instead of config files:

```python
import os
from lexilux import Chat

chat = Chat(
    base_url=os.getenv("LEXILUX_BASE_URL"),
    api_key=os.getenv("LEXILUX_API_KEY"),
    model=os.getenv("LEXILUX_MODEL"),
)
```

---

## Common Issues

### Import Error: `ModuleNotFoundError: No module named 'lexilux'`

Install lexilux in development mode:
```bash
cd /path/to/lexilux
pip install -e .
```

### Configuration File Not Found

The examples look for configuration files in this order:
1. `tests/test_endpoints.json`
2. `examples/test_endpoints.json`

If neither exists, the example will use placeholder values (which won't work with real APIs).

### Tokenizer Examples Fail

The tokenizer examples require additional dependencies:
```bash
pip install lexilux[tokenizer]
# or
pip install transformers tokenizers
```

### Vision Examples Don't Work

Multimodal examples require a vision-enabled model (like `gpt-4-vision-preview`).

---

## Learning Path

Recommended order for learning Lexilux:

1. **Start here:** `01_hello_world.py` → `02_system_message.py`
2. **Core skills:** `10_streaming.py` → `11_conversation.py` → `12_chat_params.py`
3. **Expand:** `20_embedding.py` → `21_rerank.py` → `22_tokenizer.py`
4. **Advanced:** `30_function_calling.py` → `31_multimodal.py` → `32_async.py`
5. **Expert:** `40_chat_history.py` → `41_auto_continue.py` → `42_error_handling.py` → `43_custom_formatting.py`

---

## Security Notes

⚠️ **IMPORTANT:**

1. **Never commit** `test_endpoints.json` to version control
   - It's in `.gitignore` to prevent accidental commits
   - Always verify before pushing

2. **Use different keys** for testing and production
   - Never use production keys in examples

3. **Restrict file permissions:**
   ```bash
   chmod 600 test_endpoints.json
   ```

4. **Use environment variables in production**
   - Config files are for development/testing only

---

## Additional Resources

- **Main Documentation:** [lexilux.readthedocs.io](https://lexilux.readthedocs.io)
- **GitHub Repository:** [github.com/lzjever/lexilux](https://github.com/lzjever/lexilux)
- **Issue Tracker:** [github.com/lzjever/lexilux/issues](https://github.com/lzjever/lexilux/issues)

---

## Supporting Files

| File | Purpose |
|------|---------|
| `config_loader.py` | Shared configuration loading for all examples |
| `test_endpoints.json.template` | Template for API configuration (create your own) |
| `README.md` | This file |
