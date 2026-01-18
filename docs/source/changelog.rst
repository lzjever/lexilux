Changelog
=========

.. literalinclude:: ../../CHANGELOG.md
   :language: markdown

--------------------

[2.3.0] - 2026-01-19
--------------------

Added
~~~~~

* Added comprehensive function calling (tool use) support with OpenAI-compatible API
* Added multimodal/vision support for image understanding in chat completions
* Added ``FunctionTool`` class for defining tools with JSON Schema parameters
* Added ``ToolChoice`` class for controlling when and how models use tools
* Added ``ToolCall`` dataclass for representing tool calls in responses
* Added ``ToolCallHelper`` class for high-level tool calling workflow management
* Added ``execute_tool_calls()`` helper function for executing tool calls
* Added ``create_conversation_history()`` helper for building conversation history with tools
* Added content block types: ``TextContentBlock``, ``ImageContentBlock``, ``ImageUrlDetail``
* Added support for parallel tool calls (multiple tools in one request)
* Added support for base64 encoded images in multimodal messages
* Added support for multiple images in a single request
* Added image detail level control (``low``, ``high``, ``auto``) for multimodal requests
* Added ``has_tool_calls`` property to ``ChatResult`` and ``ChatStreamChunk``
* Added ``has_content`` property to ``ChatStreamChunk`` for checking content availability
* Added comprehensive integration tests for function calling and multimodal features
* Added comprehensive unit tests for all new data types and helper functions
* Added documentation for function calling and multimodal features

Improved
~~~~~~~~

* Enhanced message normalization to support multimodal content (text + images)
* Updated type hints to support content blocks and tool calling
* Improved error handling for tool execution failures
* Updated documentation with new function calling and multimodal guides

Fixed
~~~~~

* Fixed Python 3.8-3.9 compatibility for ``NotRequired`` type
* Fixed Python 3.8-3.10 compatibility for union type syntax

[2.2.0] - 2024-XX-XX
--------------------

Added
~~~~~

* Added ``finish_reason`` field to ``ChatResult`` and ``ChatStreamChunk`` to track why generation stopped
* Added ``proxies`` parameter to ``Chat``, ``Embed``, and ``Rerank`` classes for explicit proxy configuration
* Added comprehensive integration tests for ``finish_reason`` functionality
* Added defensive handling for invalid ``finish_reason`` values from compatible services

Improved
~~~~~~~~

* Enhanced robustness of ``finish_reason`` parsing with normalization function
* Improved error handling for malformed API responses
* Updated test configuration to use new endpoint keys

Fixed
~~~~~

* Fixed test configuration key names to match updated ``test_endpoints.json`` structure
* Fixed potential issues with proxy configuration not being passed to requests

[0.1.0] - 2024-XX-XX
--------------------

Added
~~~~~

* Initial release
* Chat API support with streaming
* Embedding API support
* Rerank API support
* Tokenizer support (optional dependency on transformers)
* Unified Usage and ResultBase classes
* Comprehensive test suite
* Documentation

