"""
Chat API data models.

Defines ChatResult, ChatStreamChunk, ToolCall, and type aliases for chat completions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Sequence, Union

from lexilux.usage import Json, ResultBase, Usage

# Type aliases
Role = Literal["system", "user", "assistant", "tool"]
MessageLike = Union[str, Dict[str, str], Dict[str, Any]]
MessagesLike = Union[str, Sequence[MessageLike]]


@dataclass
class ToolCall:
    """
    Represents a function/tool call initiated by the model.

    When the model decides to call a function, it returns one or more
    ToolCall objects that specify which function to call and with what arguments.

    Attributes:
        id: Unique identifier for this tool call.
        call_id: Call reference ID (used when submitting tool outputs).
        name: Name of the function being called.
        arguments: JSON string containing function arguments.

    Examples:
        >>> tool_call = ToolCall(
        ...     id="call_abc123",
        ...     call_id="call_abc123",
        ...     name="get_weather",
        ...     arguments='{"location": "Paris", "units": "celsius"}'
        ... )
        >>> args = tool_call.get_arguments()
        >>> args
        {'location': 'Paris', 'units': 'celsius'}
    """

    id: str
    call_id: str
    name: str
    arguments: str

    def get_arguments(self) -> dict[str, Any]:
        """
        Parse and return the arguments as a dictionary.

        Returns:
            Parsed arguments dictionary.

        Raises:
            json.JSONDecodeError: If arguments string is not valid JSON.

        Examples:
            >>> tc = ToolCall(
            ...     id="call_1",
            ...     call_id="call_1",
            ...     name="get_weather",
            ...     arguments='{"location": "Paris"}'
            ... )
            >>> tc.get_arguments()
            {'location': 'Paris'}
        """
        return json.loads(self.arguments)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to API format.

        Returns:
            Dictionary in OpenAI tool call format.

        Examples:
            >>> tc = ToolCall(
            ...     id="call_1",
            ...     call_id="call_1",
            ...     name="get_weather",
            ...     arguments='{"location": "Paris"}'
            ... )
            >>> tc.to_dict()
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location": "Paris"}'}}
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class ChatResult(ResultBase):
    """
    Chat completion result (non-streaming).

    Attributes:
        text: The generated text content.
        tool_calls: List of function/tool calls initiated by the model.
        finish_reason: Reason why the generation stopped. Possible values:
            - "stop": Model stopped naturally or hit stop sequence
            - "length": Reached max_tokens limit
            - "content_filter": Content was filtered
            - "tool_calls": Model initiated tool call(s)
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
        - Tool calls: When tool_calls is non-empty, text may be empty or contain
          supplementary text alongside the function calls.

    Examples:
        >>> result = chat("Hello")
        >>> print(result.text)
        "Hello! How can I help you?"
        >>> print(result.usage.total_tokens)
        42
        >>> print(result.finish_reason)
        "stop"

        >>> # Handling tool calls:
        >>> result = chat("What's the weather in Paris?", tools=[get_weather_tool])
        >>> if result.has_tool_calls:
        ...     for tc in result.tool_calls:
        ...         print(f"Call: {tc.name} with args: {tc.get_arguments()}")

        >>> # Handling network errors:
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
        tool_calls: list[ToolCall] | None = None,
        raw: Json | None = None,
    ):
        """
        Initialize ChatResult.

        Args:
            text: Generated text content.
            usage: Usage statistics.
            finish_reason: Reason why generation stopped.
            tool_calls: List of tool calls initiated by the model.
            raw: Raw API response.
        """
        super().__init__(usage=usage, raw=raw)
        self.text = text
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        """
        Check if result contains tool calls.

        Returns:
            True if tool_calls is non-empty.

        Examples:
            >>> result = chat("...", tools=[tool])
            >>> if result.has_tool_calls:
            ...     # Handle tool calls
            ...     pass
        """
        return len(self.tool_calls) > 0

    def __str__(self) -> str:
        """Return the text content when converted to string."""
        return self.text

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ChatResult(text={self.text!r}, finish_reason={self.finish_reason!r}, usage={self.usage!r}, tool_calls={len(self.tool_calls)})"


class ChatStreamChunk(ResultBase):
    """
    Chat streaming chunk.

    Each chunk in a streaming response contains:

    - delta: The incremental text content (may be empty)
    - tool_calls: Incremental tool call data (may be empty)
    - done: Whether this is the final chunk
    - finish_reason: Reason why generation stopped (only set when done=True).
        Possible values:
        - "stop": Model stopped naturally or hit stop sequence
        - "length": Reached max_tokens limit
        - "content_filter": Content was filtered
        - "tool_calls": Model initiated tool call(s)
        - None: Still generating (intermediate chunks), [DONE] message, or unknown
    - usage: Usage statistics (may be empty/None for intermediate chunks,
      complete only in the final chunk when include_usage=True)

    Attributes:
        delta: Incremental text content.
        tool_calls: List of incremental tool call data (for streaming tool calls).
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
        - Tool calls in streaming: Tool call data is streamed incrementally.
          Multiple chunks may be needed to assemble complete tool calls.

    Examples:
        >>> for chunk in chat.stream("Hello"):
        ...     print(chunk.delta, end="")
        ...     if chunk.done:
        ...         print(f"\\nUsage: {chunk.usage.total_tokens}")
        ...         print(f"Finish reason: {chunk.finish_reason}")

        >>> # Handling tool calls in streaming:
        >>> for chunk in chat.stream("What's the weather?", tools=[tool]):
        ...     if chunk.has_tool_calls:
        ...         for tc in chunk.tool_calls:
        ...             print(f"Tool call: {tc.name}")

        >>> # Handling network errors:
        >>> try:
        ...     iterator = chat.stream("Hello")
        ...     for chunk in iterator:
        ...         if chunk.done:
        ...             break
        ... except requests.RequestException as e:
        ...     print(f"\\nNetwork error: {e}")
    """

    def __init__(
        self,
        *,
        delta: str,
        usage: Usage,
        done: bool,
        finish_reason: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        raw: Json | None = None,
    ):
        """
        Initialize ChatStreamChunk.

        Args:
            delta: Incremental text content.
            usage: Usage statistics.
            done: Whether this is the final chunk.
            finish_reason: Reason why generation stopped.
            tool_calls: List of incremental tool call data.
            raw: Raw chunk data.
        """
        super().__init__(usage=usage, raw=raw)
        self.delta = delta
        self.done = done
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls or []

    @property
    def has_content(self) -> bool:
        """
        Check if chunk contains text content.

        Returns:
            True if delta is non-empty.

        Examples:
            >>> chunk = ChatStreamChunk(delta="Hello", usage=Usage(), done=False)
            >>> chunk.has_content
            True
        """
        return bool(self.delta)

    @property
    def has_tool_calls(self) -> bool:
        """
        Check if chunk contains tool call data.

        Returns:
            True if tool_calls is non-empty.

        Examples:
            >>> chunk = ChatStreamChunk(
            ...     delta="",
            ...     usage=Usage(),
            ...     done=False,
            ...     tool_calls=[ToolCall(...)]
            ... )
            >>> chunk.has_tool_calls
            True
        """
        return len(self.tool_calls) > 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ChatStreamChunk(delta={self.delta!r}, done={self.done}, finish_reason={self.finish_reason!r}, usage={self.usage!r}, tool_calls={len(self.tool_calls)})"
