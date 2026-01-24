"""
Streaming result accumulation.

Provides StreamingResult and StreamingIterator for automatic text accumulation
during streaming, allowing history to be updated in real-time.

Also provides async versions: AsyncStreamingIterator for async streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.usage import Usage

if TYPE_CHECKING:
    pass


class StreamingResult:
    """
    Streaming accumulated result (can be used as ChatResult).

    Automatically accumulates text during streaming, content updates automatically
    on each iteration. Can be used as a string, or converted to ChatResult.
    """

    __slots__ = ("_text_parts", "_text_cache", "_finish_reason", "_usage", "_done")

    def __init__(self) -> None:
        """Initialize accumulated result."""
        # Use list buffer for O(n) instead of O(n²) string concatenation
        self._text_parts: list[str] = []
        self._text_cache: str | None = None  # Lazy cache for joined text
        self._finish_reason: str | None = None
        self._usage: Usage = Usage()
        self._done: bool = False

    def update(self, chunk: ChatStreamChunk) -> None:
        """Update accumulated content (internal call)."""
        if chunk.delta:
            self._text_parts.append(chunk.delta)
            self._text_cache = None  # Invalidate cache
        if chunk.done:
            self._done = True
            # Only update finish_reason if chunk provides one (don't overwrite with None)
            if chunk.finish_reason is not None:
                self._finish_reason = chunk.finish_reason
            if chunk.usage:
                self._usage = chunk.usage

    @property
    def text(self) -> str:
        """Get currently accumulated text (can be used as string)."""
        if self._text_cache is None:
            self._text_cache = "".join(self._text_parts)
        return self._text_cache

    @property
    def finish_reason(self) -> str | None:
        """Get finish_reason."""
        return self._finish_reason

    @property
    def usage(self) -> Usage:
        """Get usage."""
        return self._usage

    @property
    def done(self) -> bool:
        """Whether streaming is done."""
        return self._done

    def to_chat_result(self) -> ChatResult:
        """Convert to ChatResult (for history)."""
        return ChatResult(
            text=self.text,
            finish_reason=self._finish_reason,
            usage=self._usage,
        )

    def __str__(self) -> str:
        """Use as string."""
        return self.text

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"StreamingResult(text={self.text!r}, done={self._done}, "
            f"finish_reason={self._finish_reason!r})"
        )


class StreamingIterator:
    """
    Streaming iterator (wraps original iterator, provides accumulated result).

    Automatically updates accumulated result on each iteration, user can access
    current state at any time.
    """

    def __init__(self, chunk_iterator: Iterator[ChatStreamChunk]) -> None:
        """Initialize."""
        self._iterator = chunk_iterator
        self._result = StreamingResult()

    def __iter__(self) -> Iterator[ChatStreamChunk]:
        """Iterate chunks."""
        for chunk in self._iterator:
            self._result.update(chunk)  # Auto-accumulate
            yield chunk

    @property
    def result(self) -> StreamingResult:
        """Get currently accumulated result (accessible at any time)."""
        return self._result


class AsyncStreamingIterator:
    """
    Async streaming iterator (wraps async iterator, provides accumulated result).

    Automatically updates accumulated result on each iteration, user can access
    current state at any time.

    Examples:
        >>> async for chunk in chat.astream("Hello"):
        ...     print(chunk.delta, end="")
        >>> result = iterator.result.to_chat_result()
    """

    def __init__(self, chunk_iterator: AsyncIterator[ChatStreamChunk]) -> None:
        """
        Initialize async streaming iterator.

        Args:
            chunk_iterator: Async iterator yielding ChatStreamChunk objects.
        """
        self._iterator = chunk_iterator
        self._result = StreamingResult()

    def __aiter__(self) -> AsyncIterator[ChatStreamChunk]:
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> ChatStreamChunk:
        """Get next chunk and update accumulated result."""
        try:
            chunk = await self._iterator.__anext__()
            self._result.update(chunk)
            return chunk
        except StopAsyncIteration:
            raise

    @property
    def result(self) -> StreamingResult:
        """Get currently accumulated result (accessible at any time)."""
        return self._result

    async def collect(self) -> ChatResult:
        """
        Consume all chunks and return final ChatResult.

        This is a convenience method to consume the entire stream
        and get the final result.

        Returns:
            ChatResult with accumulated text and usage.

        Examples:
            >>> result = await chat.astream("Hello").collect()
            >>> print(result.text)
        """
        async for _ in self:
            pass
        return self._result.to_chat_result()
