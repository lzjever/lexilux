Introduction
============

Lexilux is a unified LLM API client library that provides a simple, function-like API for interacting with Chat, Embedding, Rerank, and Tokenizer services.

Key Features
------------

* **Function-like API**: Call APIs like functions (``chat("hi")``, ``embed(["text"])``)
* **Streaming Support**: Built-in streaming for Chat with usage tracking
* **Unified Usage**: Consistent usage statistics across all APIs
* **Flexible Input**: Support multiple input formats (string, list, dict)
* **OpenAI-Compatible**: Works with OpenAI-compatible APIs
* **Automatic Retry**: Built-in retry logic with exponential backoff
* **Connection Pooling**: HTTP connection pooling for better performance
* **Exception Hierarchy**: Comprehensive exception system with error codes
* **Function Calling**: OpenAI-compatible function/tool calling support
* **Multimodal Support**: Vision capabilities with image inputs
* **Async Support**: Full async/await API for concurrent operations

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
* :doc:`async` - Learn about async/await support
* :doc:`api_reference/index` - Complete API reference
