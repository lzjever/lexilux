"""
Chat API data models.

Defines ChatResult, ChatStreamChunk, and type aliases for chat completions.
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

from lexilux.usage import Json, ResultBase, Usage

# Type aliases
Role = Literal["system", "user", "assistant", "tool"]
MessageLike = Union[str, dict[str, str]]
MessagesLike = Union[str, Sequence[MessageLike]]


class ChatResult(ResultBase):
    """
    Chat completion result (non-streaming).

    Attributes:
        text: The generated text content.
        finish_reason: Reason why the generation stopped. Possible values:
            - "stop": Model stopped naturally or hit stop sequence
            - "length": Reached max_tokens limit
            - "content_filter": Content was filtered
            - None: Unknown or not provided
        usage: Usage statistics.
        raw: Raw API response.

    Important Notes:
        - finish_reason is only available when the API successfully returns a response.
        - If network connection is interrupted, an exception will be raised
          (requests.RequestException, ConnectionError, TimeoutError, etc.)
          and no ChatResult will be returned.
        - To distinguish network errors from normal completion:
          * Network error: Exception is raised, no ChatResult returned
          * Normal completion: ChatResult returned with finish_reason set

    Examples:
        >>> result = chat("Hello")
        >>> print(result.text)
        "Hello! How can I help you?"
        >>> print(result.usage.total_tokens)
        42
        >>> print(result.finish_reason)
        "stop"

        Handling network errors:
        >>> try:
        ...     result = chat("Hello")
        ...     print(f"Finished: {result.finish_reason}")
        ... except requests.RequestException as e:
        ...     print(f"Network error: {e}")
        ...     # No finish_reason available - connection failed
    """

    def __init__(
        self,
        *,
        text: str,
        usage: Usage,
        finish_reason: str | None = None,
        raw: Json | None = None,
    ):
        """
        Initialize ChatResult.

        Args:
            text: Generated text content.
            usage: Usage statistics.
            finish_reason: Reason why generation stopped.
            raw: Raw API response.
        """
        super().__init__(usage=usage, raw=raw)
        self.text = text
        self.finish_reason = finish_reason

    def __str__(self) -> str:
        """Return the text content when converted to string."""
        return self.text

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ChatResult(text={self.text!r}, finish_reason={self.finish_reason!r}, usage={self.usage!r})"


class ChatStreamChunk(ResultBase):
    """
    Chat streaming chunk.

    Each chunk in a streaming response contains:

    - delta: The incremental text content (may be empty)
    - done: Whether this is the final chunk
    - finish_reason: Reason why generation stopped (only set when done=True).
        Possible values:
        - "stop": Model stopped naturally or hit stop sequence
        - "length": Reached max_tokens limit
        - "content_filter": Content was filtered
        - None: Still generating (intermediate chunks), [DONE] message, or unknown
    - usage: Usage statistics (may be empty/None for intermediate chunks,
      complete only in the final chunk when include_usage=True)

    Attributes:
        delta: Incremental text content.
        done: Whether this is the final chunk.
        finish_reason: Reason why generation stopped (None for intermediate chunks).
        usage: Usage statistics (may be incomplete for intermediate chunks).
        raw: Raw chunk data.

    Important Notes:
        - finish_reason is only available when the API successfully completes.
        - If network connection is interrupted, an exception will be raised
          (requests.RequestException, ConnectionError, TimeoutError, etc.)
          and no chunk with finish_reason will be received.
        - To distinguish network errors from normal completion:
          * Network error: Exception is raised, no done=True chunk received
          * Normal completion: done=True chunk received with finish_reason set
          * Incomplete stream: Exception raised after receiving some chunks
            (check if any chunk has done=True to determine if completion occurred)

    Examples:
        >>> for chunk in chat.stream("Hello"):
        ...     print(chunk.delta, end="")
        ...     if chunk.done:
        ...         print(f"\\nUsage: {chunk.usage.total_tokens}")
        ...         print(f"Finish reason: {chunk.finish_reason}")

        Handling network errors:
        >>> try:
        ...     for chunk in chat.stream("Hello"):
        ...         print(chunk.delta, end="")
        ...         if chunk.done:
        ...             print(f"\\nFinished: {chunk.finish_reason}")
        ... except requests.RequestException as e:
        ...     print(f"\\nNetwork error: {e}")
        ...     # No finish_reason available - connection was interrupted
    """

    def __init__(
        self,
        *,
        delta: str,
        done: bool,
        usage: Usage,
        finish_reason: str | None = None,
        raw: Json | None = None,
    ):
        """
        Initialize ChatStreamChunk.

        Args:
            delta: Incremental text content.
            done: Whether this is the final chunk.
            usage: Usage statistics.
            finish_reason: Reason why generation stopped.
            raw: Raw chunk data.
        """
        super().__init__(usage=usage, raw=raw)
        self.delta = delta
        self.done = done
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ChatStreamChunk(delta={self.delta!r}, done={self.done}, finish_reason={self.finish_reason!r}, usage={self.usage!r})"
