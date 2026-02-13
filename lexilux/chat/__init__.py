"""
Chat API module.

Provides Chat client, result models, parameter configuration, and tool support for chat completions.
"""

import warnings

from lexilux.chat.client import Chat
from lexilux.chat.content_blocks import (
    ContentBlock,
    ImageContentBlock,
    ImageUrlDetail,
    TextContentBlock,
)
from lexilux.chat.conversation import _ResponseContinuer
from lexilux.chat.exceptions import (
    ChatIncompleteResponseError,
    ChatStreamInterruptedError,
)
from lexilux.chat.formatters import ChatHistoryFormatter
from lexilux.chat.history import (
    ChatHistory,
    TokenAnalysis,
    filter_by_role,
    get_statistics,
    merge_histories,
    search_content,
)
from lexilux.chat.models import (
    ChatResult,
    ChatStreamChunk,
    MessageLike,
    MessagesLike,
    Role,
    StreamingToolCall,
    ToolCall,
)
from lexilux.chat.params import ChatParams
from lexilux.chat.streaming import (
    AsyncStreamingIterator,
    StreamingIterator,
    StreamingResult,
)
from lexilux.chat.tool_helpers import (
    ToolCallHelper,
    create_conversation_history,
    execute_tool_calls,
)
from lexilux.chat.tools import FunctionTool, Tool, ToolChoice
from lexilux.chat.types import (
    ChatResponse,
    ChatResponseChoice,
    ContinuePromptCallable,
    ErrorCallback,
    JSONValue,
    JsonObject,
    MessageDict,
    ProgressCallback,
    ToolCallDict,
    UsageDict,
)
from lexilux.chat.utils import normalize_messages


# Backward compatibility aliases with deprecation warnings
def __getattr__(name: str):
    if name == "ChatContinue":
        warnings.warn(
            "ChatContinue is deprecated and will be removed in v3.0.0. "
            "Use chat.complete() for automatic continuation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ResponseContinuer
    if name == "Conversation":
        warnings.warn(
            "Conversation is deprecated and will be removed in v3.0.0. "
            "Use chat.complete() for automatic continuation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ResponseContinuer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main classes
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
    # Types
    "Role",
    "MessageLike",
    "MessagesLike",
    "ToolCall",
    "StreamingToolCall",
    "Tool",
    "FunctionTool",
    "ToolChoice",
    # Content blocks
    "ContentBlock",
    "TextContentBlock",
    "ImageContentBlock",
    "ImageUrlDetail",
    # Tool helpers
    "ToolCallHelper",
    "execute_tool_calls",
    "create_conversation_history",
    # Utility functions
    "normalize_messages",
    "merge_histories",
    "filter_by_role",
    "search_content",
    "get_statistics",
    # Exceptions
    "ChatStreamInterruptedError",
    "ChatIncompleteResponseError",
    # Type aliases
    "JSONValue",
    "JsonObject",
    "MessageDict",
    "ToolCallDict",
    "UsageDict",
    "ChatResponseChoice",
    "ChatResponse",
    "ContinuePromptCallable",
    "ProgressCallback",
    "ErrorCallback",
]
