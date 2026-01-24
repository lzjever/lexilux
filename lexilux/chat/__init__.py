"""
Chat API module.

Provides Chat client, result models, parameter configuration, and tool support for chat completions.
"""

from lexilux.chat.client import Chat
from lexilux.chat.content_blocks import (
    ContentBlock,
    ImageContentBlock,
    ImageUrlDetail,
    TextContentBlock,
)
from lexilux.chat.conversation import Conversation
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
from lexilux.chat.utils import normalize_messages

# Backward compatibility alias
ChatContinue = Conversation

__all__ = [
    # Main classes
    "Chat",
    "ChatResult",
    "ChatStreamChunk",
    "ChatParams",
    "ChatHistory",
    "ChatHistoryFormatter",
    "Conversation",
    "ChatContinue",  # Backward compatibility alias
    "StreamingResult",
    "StreamingIterator",
    "AsyncStreamingIterator",
    "TokenAnalysis",
    # Types
    "Role",
    "MessageLike",
    "MessagesLike",
    "ToolCall",
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
]
