# Lexilux 🚀

[![PyPI version](https://img.shields.io/pypi/v/lexilux.svg)](https://pypi.org/project/lexilux/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/lexilux/badge/?version=latest)](https://lexilux.readthedocs.io)
[![CI](https://github.com/lzjever/lexilux/workflows/CI/badge.svg)](https://github.com/lzjever/lexilux/actions)
[![codecov](https://codecov.io/gh/lzjever/lexilux/branch/main/graph/badge.svg)](https://codecov.io/gh/lzjever/lexilux)

**Lexilux** is a unified LLM API client library that makes calling Chat, Embedding, Rerank, and Tokenizer APIs as simple as calling a function.

## ✨ Features

- 🎯 **Function-like API**: Call APIs like functions (`chat("hi")`, `embed(["text"])`)
- 🔄 **Streaming Support**: Built-in streaming for Chat with usage tracking
- 📊 **Unified Usage**: Consistent usage statistics across all APIs
- 🔧 **Flexible Input**: Support multiple input formats (string, list, dict)
- 🚫 **Optional Dependencies**: Tokenizer requires transformers only when needed
- 🌐 **OpenAI-Compatible**: Works with OpenAI-compatible APIs

## 📦 Installation

### Quick Install

```bash
pip install lexilux
```

### With Tokenizer Support

```bash
# Using full name
pip install lexilux[tokenizer]

```

### Development Setup with uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management. Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the development environment:

**Recommended: For active development**

```bash
# Install package with all development dependencies (recommended)
make dev-install

# Or manually with uv (dev group is installed by default)
uv sync --group docs --all-extras
```

**Alternative: Dependencies only (for CI/CD or code review)**

```bash
# Create virtual environment and install dependencies only (without installing the package)
# Useful for: CI/CD pipelines, code review, or when you only need development tools
make setup-venv

# Later, if you need to install the package:
make install
```

**Understanding dependency groups vs extras:**

- **Dependency groups** (`dev`, `docs`): Development dependencies that are not published to PyPI. The `dev` group is installed by default with `uv sync`.
- **Extras** (`tokenizer`, `token`): Optional runtime features for tokenizer support - install with `--extra tokenizer` or `--all-extras`.

All `make` commands will automatically use `uv` if available, otherwise fall back to `pip`.

### Development Install (Legacy - using pip)

For development with all dependencies using pip:

```bash
pip install -e ".[dev]"
# Or using Makefile
make dev-install
```

## 🚀 Quick Start

### Chat

```python
from lexilux import Chat

chat = Chat(base_url="https://api.example.com/v1", api_key="your-key", model="gpt-4")

# Simple call
result = chat("Hello, world!")
print(result.text)  # "Hello! How can I help you?"
print(result.usage.total_tokens)  # 42

# With system message
result = chat("What is Python?", system="You are a helpful assistant")

# Streaming
for chunk in chat.stream("Tell me a joke"):
    print(chunk.delta, end="")
    if chunk.done:
        print(f"\nUsage: {chunk.usage.total_tokens}")
```

### Embedding

```python
from lexilux import Embed

embed = Embed(base_url="https://api.example.com/v1", api_key="your-key", model="text-embedding-ada-002")

# Single text
result = embed("Hello, world!")
vector = result.vectors  # List[float]

# Batch
result = embed(["text1", "text2"])
vectors = result.vectors  # List[List[float]]
```

### Rerank

```python
from lexilux import Rerank

# OpenAI-compatible mode (default)
rerank = Rerank(
    base_url="https://api.example.com/v1", 
    api_key="your-key", 
    model="rerank-model",
    mode="openai"  # or "dashscope" for DashScope API
)

result = rerank("python http", ["urllib", "requests", "httpx"])
ranked = result.results  # List[Tuple[int, float]] - (index, score)

# With documents included
result = rerank("query", ["doc1", "doc2"], include_docs=True)
ranked = result.results  # List[Tuple[int, float, str]] - (index, score, doc)
```

### Tokenizer

```python
from lexilux import Tokenizer

# Auto-offline mode (use local cache if available, download if not)
tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", mode="auto_offline")

result = tokenizer("Hello, world!")
print(result.usage.input_tokens)  # 3
print(result.input_ids)  # [[15496, 11, 1917, 0]]

# Force-offline mode (for air-gapped environments)
tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", mode="force_offline", cache_dir="/models/hf")
```

## 📚 Documentation

Full documentation available at: [lexilux.readthedocs.io](https://lexilux.readthedocs.io)

## 📖 Examples

Check out the `examples/` directory for practical examples:

- **`basic_chat.py`** - Simple chat completion
- **`chat_streaming.py`** - Streaming chat
- **`embedding_demo.py`** - Text embedding
- **`rerank_demo.py`** - Document reranking
- **`tokenizer_demo.py`** - Tokenization

Run examples:

```bash
python examples/basic_chat.py
```

## 🧪 Testing

```bash
# Run all unit tests (excludes integration tests)
make test

# Run integration tests (requires external services)
make test-integration

# Run with coverage
make test-cov

# Run linting
make lint

# Format code
make format
```

## 📚 Documentation

Full documentation available at: [lexilux.readthedocs.io](https://lexilux.readthedocs.io)

Build documentation locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

## 🏢 About Agentsmith

**Lexilux** is part of the **Agentsmith** open-source ecosystem. Agentsmith is a ToB AI agent and algorithm development platform, currently deployed in multiple highway management companies, securities firms, and regulatory agencies in China. The Agentsmith team is gradually open-sourcing the platform by removing proprietary code and algorithm modules, as well as enterprise-specific customizations, while decoupling the system for modular use by the open-source community.

### 🌟 Agentsmith Open-Source Projects

- **[Varlord](https://github.com/lzjever/varlord)** ⚙️ - Configuration management library with multi-source support
- **[Routilux](https://github.com/lzjever/routilux)** ⚡ - Event-driven workflow orchestration framework
- **[Serilux](https://github.com/lzjever/serilux)** 📦 - Flexible serialization framework for Python objects
- **[Lexilux](https://github.com/lzjever/lexilux)** 🚀 - Unified LLM API client library

These projects are modular components extracted from the Agentsmith platform, designed to be used independently or together to build powerful applications.


## 📄 License

Lexilux is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

## 🔗 Links

- **📦 PyPI**: [pypi.org/project/lexilux](https://pypi.org/project/lexilux)
- **📚 Documentation**: [lexilux.readthedocs.io](https://lexilux.readthedocs.io)
- **🐙 GitHub**: [github.com/lzjever/lexilux](https://github.com/lzjever/lexilux)

---

**Built with ❤️ by the Lexilux Team**

