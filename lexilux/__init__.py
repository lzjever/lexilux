"""
Lexilux - Unified LLM API client library

Provides Chat, Embedding, Rerank, and Tokenizer support with a simple, function-like API.
"""

from lexilux.chat import (
    Chat,
    ChatContinue,
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
from lexilux.tokenizer import Tokenizer, TokenizeResult
from lexilux.usage import ResultBase, Usage

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
    "ChatContinue",
    "StreamingResult",
    "StreamingIterator",
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
]

__version__ = "2.2.0"
