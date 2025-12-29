"""
Lexilux - Unified LLM API client library

Provides Chat, Embedding, Rerank, and Tokenizer support with a simple, function-like API.
"""

from lexilux.usage import Usage, ResultBase
from lexilux.chat import Chat, ChatResult, ChatStreamChunk
from lexilux.embed import Embed, EmbedResult
from lexilux.rerank import Rerank, RerankResult
from lexilux.tokenizer import Tokenizer, TokenizeResult, TokenizerMode

__version__ = "0.1.2"

__all__ = [
    # Usage
    "Usage",
    "ResultBase",
    # Chat
    "Chat",
    "ChatResult",
    "ChatStreamChunk",
    # Embed
    "Embed",
    "EmbedResult",
    # Rerank
    "Rerank",
    "RerankResult",
    # Tokenizer
    "Tokenizer",
    "TokenizeResult",
    "TokenizerMode",
]
