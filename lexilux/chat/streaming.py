"""
Streaming result accumulation.

Provides StreamingResult and StreamingIterator for automatic text accumulation
during streaming, allowing history to be updated in real-time.
"""

from __future__ import annotations

from typing import Iterator

from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.usage import Usage


class StreamingResult:
    """
    Streaming accumulated result (can be used as ChatResult).

    Automatically accumulates text during streaming, content updates automatically
    on each iteration. Can be used as a string, or converted to ChatResult.
    """

    def __init__(self) -> None:
        """Initialize accumulated result."""
        self._text: str = ""
        self._finish_reason: str | None = None
        self._usage: Usage = Usage()
        self._done: bool = False

    def update(self, chunk: ChatStreamChunk) -> None:
        """Update accumulated content (internal call)."""
        self._text += chunk.delta
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
        return self._text

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
            text=self._text,
            finish_reason=self._finish_reason,
            usage=self._usage,
        )

    def __str__(self) -> str:
        """Use as string."""
        return self._text

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"StreamingResult(text={self._text!r}, done={self._done}, "
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
