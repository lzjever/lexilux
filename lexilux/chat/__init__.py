"""
Chat API module.

Provides Chat client, result models, and parameter configuration for chat completions.
"""

from lexilux.chat.client import Chat
from lexilux.chat.formatters import ChatHistoryFormatter
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike, Role
from lexilux.chat.params import ChatParams
from lexilux.chat.streaming import StreamingIterator, StreamingResult

__all__ = [
    "Chat",
    "ChatResult",
    "ChatStreamChunk",
    "ChatParams",
    "ChatHistory",
    "ChatHistoryFormatter",
    "StreamingResult",
    "StreamingIterator",
    "Role",
    "MessageLike",
    "MessagesLike",
]
