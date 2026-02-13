"""
Lexilux - Unified LLM API client library

Provides Chat, Embedding, Rerank, and Tokenizer support with a simple, function-like API.
"""

import warnings

from lexilux.chat import (
    AsyncStreamingIterator,
    Chat,
    ChatHistory,
    ChatHistoryFormatter,
    ChatParams,
    ChatResult,
    ChatStreamChunk,
    ContentBlock,
    FunctionTool,
    ImageContentBlock,
    ImageUrlDetail,
    StreamingIterator,
    StreamingResult,
    TextContentBlock,
    TokenAnalysis,
    Tool,
    ToolCall,
    ToolCallHelper,
    ToolChoice,
    create_conversation_history,
    execute_tool_calls,
    filter_by_role,
    get_statistics,
    merge_histories,
    normalize_messages,
    search_content,
)
from lexilux.chat.conversation import _ResponseContinuer
from lexilux.embed import Embed, EmbedResult
from lexilux.embed_params import EmbedParams
from lexilux.exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    InvalidRequestError,
    LexiluxError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from lexilux.rerank import Rerank, RerankResult
from lexilux.registry import (
    ChatFactory,
    ConfiguredChat,
    ModelCapabilities,
    ModelCost,
    ModelLimits,
    ModelModalities,
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)
from lexilux.tokenizer import Tokenizer, TokenizeResult
from lexilux.usage import ResultBase, Usage


# Backward compatibility aliases with deprecation warnings
def __getattr__(name: str):
    """Provide backward-compatible access to deprecated names."""
    if name == "Conversation":
        warnings.warn(
            "Conversation is deprecated and will be removed in v3.0.0. "
            "Use chat.complete() for automatic continuation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ResponseContinuer
    if name == "ChatContinue":
        warnings.warn(
            "ChatContinue is deprecated and will be removed in v3.0.0. "
            "Use chat.complete() for automatic continuation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ResponseContinuer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Exceptions
    "LexiluxError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "NotFoundError",
    "ServerError",
    "NetworkError",
    "TimeoutError",
    "ConnectionError",
    "ValidationError",
    "ConfigurationError",
    # Usage
    "Usage",
    "ResultBase",
    # Chat
    "Chat",
    "ChatResult",
    "ChatStreamChunk",
    "ChatParams",
    "ChatHistory",
    "ChatHistoryFormatter",
    "StreamingResult",
    "StreamingIterator",
    "AsyncStreamingIterator",
    "TokenAnalysis",
    # Tool calling and multimodal
    "Tool",
    "FunctionTool",
    "ToolChoice",
    "ToolCall",
    "ToolCallHelper",
    "ContentBlock",
    "TextContentBlock",
    "ImageContentBlock",
    "ImageUrlDetail",
    # Tool helpers
    "execute_tool_calls",
    "create_conversation_history",
    "normalize_messages",
    # Utility functions
    "merge_histories",
    "filter_by_role",
    "search_content",
    "get_statistics",
    # Embed
    "Embed",
    "EmbedResult",
    "EmbedParams",
    # Rerank
    "Rerank",
    "RerankResult",
    # Tokenizer
    "Tokenizer",
    "TokenizeResult",
    # Registry
    "ModelRegistry",
    "ChatFactory",
    "ConfiguredChat",
    "ModelSpec",
    "ProviderSpec",
    "ModelCapabilities",
    "ModelLimits",
    "ModelModalities",
    "ModelCost",
]

__version__ = "2.7.2"
