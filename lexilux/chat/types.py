"""
Type definitions for chat module.

This module provides type aliases and TypedDicts for better type safety
while maintaining flexibility for API responses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, Union

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from lexilux.chat.models import ChatResult


# JSON value types - recursive type for any valid JSON value
# Note: Using Any for the recursive parts due to mypy limitations
JSONValue: TypeAlias = Union[
    str, int, float, bool, None, dict[str, "JSONValue"], list["JSONValue"]
]

# Type alias for JSON objects
JsonObject: TypeAlias = dict[str, JSONValue]


class MessageDict(TypedDict, total=False):
    """Type for a single message in the chat API."""

    role: str
    content: str | list[JsonObject] | None
    name: str | None
    tool_calls: list[JsonObject] | None
    tool_call_id: str | None


class ToolCallDict(TypedDict, total=False):
    """Type for a tool call in the API response."""

    id: str
    type: str
    function: dict[str, str]


class UsageDict(TypedDict, total=False):
    """Type for usage statistics in API response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: JsonObject | None
    completion_tokens_details: JsonObject | None


class ChatResponseChoice(TypedDict, total=False):
    """Type for a choice in chat completion response."""

    index: int
    message: MessageDict
    delta: MessageDict
    finish_reason: str | None


class ChatResponse(TypedDict, total=False):
    """Type for the full chat completion API response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatResponseChoice]
    usage: UsageDict


# Continue callback types
ContinuePromptCallable: TypeAlias = Callable[[int, int, str, str], str]
"""
Type for continue_prompt callable.

Args:
    continue_count: Current continuation count (1-indexed)
    max_continues: Maximum number of continuations allowed
    current_text: Text accumulated so far
    original_prompt: Original user prompt (if available)

Returns:
    The prompt string to use for continuation
"""

ProgressCallback: TypeAlias = Callable[
    [int, int, "ChatResult", list["ChatResult"]], None
]
"""
Type for on_progress callback.

Args:
    continue_count: Current continuation count
    max_continues: Maximum number of continuations
    current_result: The most recent ChatResult
    all_results: All ChatResult objects collected so far
"""

ErrorCallback: TypeAlias = Callable[
    [Exception, "ChatResult"], Union[dict[str, Union[str, "ChatResult"]], None]
]
"""
Type for on_error_callback.

Args:
    error: The exception that occurred
    partial_result: The partial result before the error

Returns:
    Optional dict with 'action' key ('raise', 'return_partial', 'retry')
    and optional 'result' key with a ChatResult
"""
