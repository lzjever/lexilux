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
    StreamingIterator,
    StreamingResult,
    TokenAnalysis,
    filter_by_role,
    get_statistics,
    merge_histories,
    search_content,
)
from lexilux.embed import Embed, EmbedResult
from lexilux.embed_params import EmbedParams
from lexilux.rerank import Rerank, RerankResult
from lexilux.tokenizer import Tokenizer, TokenizeResult
from lexilux.usage import ResultBase, Usage

__all__ = [
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

__version__ = "2.1.0"
