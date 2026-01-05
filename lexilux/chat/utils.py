"""
Utility functions for chat API.

Provides message normalization, finish_reason normalization, and usage parsing.
"""

from __future__ import annotations

from typing import Any

from lexilux.chat.models import MessagesLike
from lexilux.usage import Json, Usage


def normalize_messages(
    messages: MessagesLike,
    system: str | None = None,
) -> list[dict[str, str]]:
    """
    Normalize messages input to a list of message dictionaries.

    Supports multiple input formats:
    - str: Converted to [{"role": "user", "content": str}]
    - List[Dict]: Used as-is
    - List[str]: Converted to [{"role": "user", "content": str}, ...]

    Args:
        messages: Messages in various formats.
        system: Optional system message to prepend.

    Returns:
        Normalized list of message dictionaries.

    Examples:
        >>> normalize_messages("hi")
        [{"role": "user", "content": "hi"}]

        >>> normalize_messages([{"role": "user", "content": "hi"}])
        [{"role": "user", "content": "hi"}]

        >>> normalize_messages("hi", system="You are helpful")
        [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "hi"}]
    """
    result: list[dict[str, str]] = []

    # Add system message if provided
    if system:
        result.append({"role": "system", "content": system})

    # Normalize messages
    if isinstance(messages, str):
        # Single string -> single user message
        result.append({"role": "user", "content": messages})
    elif isinstance(messages, (list, tuple)):
        # List of messages
        for msg in messages:
            if isinstance(msg, str):
                # String in list -> user message
                result.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                # Dict -> use as-is (should have "role" and "content")
                if "role" in msg and "content" in msg:
                    result.append({"role": msg["role"], "content": msg["content"]})
                else:
                    raise ValueError(
                        f"Invalid message dict: {msg}. Must have 'role' and 'content' keys."
                    )
            else:
                raise ValueError(f"Invalid message type: {type(msg)}. Expected str or dict.")
    else:
        raise ValueError(f"Invalid messages type: {type(messages)}. Expected str, list, or tuple.")

    return result


def parse_usage(response_data: Json) -> Usage:
    """
    Parse usage information from API response.

    Args:
        response_data: API response data.

    Returns:
        Usage object.
    """
    usage_data = response_data.get("usage")
    if usage_data is None:
        usage_data = {}
    elif not isinstance(usage_data, dict):
        usage_data = {}

    return Usage(
        input_tokens=usage_data.get("prompt_tokens") or usage_data.get("input_tokens"),
        output_tokens=usage_data.get("completion_tokens") or usage_data.get("output_tokens"),
        total_tokens=usage_data.get("total_tokens"),
        details=usage_data,
    )


def normalize_finish_reason(finish_reason: Any) -> str | None:
    """
    Normalize finish_reason to a valid string or None.

    Handles cases where compatible services may return invalid values:
    - None -> None
    - Empty string "" -> None
    - Valid string ("stop", "length", "content_filter") -> as-is
    - Other types (int, bool, etc.) -> None (defensive)

    Args:
        finish_reason: Raw finish_reason value from API.

    Returns:
        Normalized finish_reason (str or None).
    """
    if finish_reason is None:
        return None
    if isinstance(finish_reason, str):
        # Empty string should be treated as None
        return finish_reason if finish_reason else None
    # For any other type (int, bool, list, etc.), return None defensively
    return None
