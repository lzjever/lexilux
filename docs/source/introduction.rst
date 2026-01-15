Introduction
============

Lexilux is a unified LLM API client library that provides a simple, function-like API for interacting with Chat, Embedding, Rerank, and Tokenizer services.

Key Features
------------

* **Function-like API**: Call APIs like functions (``chat("hi")``, ``embed(["text"])``)
* **Streaming Support**: Built-in streaming for Chat with automatic text accumulation
* **Explicit History Management**: Full control over conversation history with explicit ``ChatHistory`` objects
* **Continue Generation**: Seamlessly continue cut-off responses with ``ChatContinue``
* **Function Calling**: OpenAI-compatible function/tool calling with helper utilities
* **Multimodal Support**: Vision capabilities with image inputs (URL and base64)
* **Comprehensive Token Analysis**: Detailed token statistics and analysis for conversation history
* **Unified Usage**: Consistent usage statistics across all APIs
* **Flexible Input**: Support multiple input formats (string, list, dict)
* **History Management**: Automatic extraction, serialization, and multi-format export
* **Optional Dependencies**: Tokenizer requires transformers only when needed
* **OpenAI-Compatible**: Works with OpenAI-compatible APIs
* **Automatic Retry**: Built-in retry logic with exponential backoff for transient failures
* **Connection Pooling**: HTTP connection pooling for better performance under high concurrency
* **Exception Hierarchy**: Comprehensive exception system with error codes and retryable flags

Design Philosophy
-----------------

Lexilux is designed to be as simple as possible while remaining powerful and flexible. The API follows these principles:

1. **Simplicity**: Call APIs like functions, no complex setup required
2. **Consistency**: All APIs return results with unified usage statistics
3. **Flexibility**: Support multiple input formats and optional parameters
4. **Extensibility**: Easy to extend and customize for different use cases
5. **Reliability**: Built-in error handling, retry logic, and connection pooling

What's Next?
------------

* :doc:`installation` - Install Lexilux
* :doc:`quickstart` - Get started in minutes
* :doc:`function_calling` - Learn about function calling with tools
* :doc:`api_reference/index` - Complete API reference

