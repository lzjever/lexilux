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
) -> list[dict[str, Any]]:
    """
    Normalize messages input to a list of message dictionaries.

    Supports multiple input formats with backward compatibility:
    - str: Converted to [{"role": "user", "content": str}]
    - List[Dict[str, str]]: Used as-is (legacy format, content is string)
    - List[Dict[str, Any]]: Used as-is (supports multimodal content as list)
    - List[str]: Converted to [{"role": "user", "content": str}, ...]

    Multimodal content is supported by passing content as a list of blocks:
    [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]

    Args:
        messages: Messages in various formats.
        system: Optional system message to prepend.

    Returns:
        Normalized list of message dictionaries.

    Examples:
        >>> # Simple string
        >>> normalize_messages("hi")
        [{'role': 'user', 'content': 'hi'}]

        >>> # Legacy format (content as string)
        >>> normalize_messages([{"role": "user", "content": "hi"}])
        [{'role': 'user', 'content': 'hi'}]

        >>> # Multimodal format
        >>> normalize_messages([{
        ...     "role": "user",
        ...     "content": [
        ...         {"type": "text", "text": "What's in this image?"},
        ...         {"type": "image_url", "image_url": {"url": "https://..."}}
        ...     ]
        ... }])

        >>> # With system message
        >>> normalize_messages("hi", system="You are helpful")
        [{'role': 'system', 'content': 'You are helpful'}, {'role': 'user', 'content': 'hi'}]
    """
    result: list[dict[str, Any]] = []

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
                # Dict -> validate and use as-is
                if "role" not in msg:
                    raise ValueError(
                        f"Invalid message dict: {msg}. Must have 'role' key."
                    )

                # Allow assistant messages with tool_calls to omit content
                # (OpenAI API allows content to be null/omitted when tool_calls exist)
                if "content" not in msg:
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        content = None
                    else:
                        raise ValueError(
                            f"Invalid message dict: {msg}. Must have 'content' key."
                        )
                else:
                    content = msg["content"]
                # Validate content format (skip for None when tool_calls exist)
                if content is not None and not isinstance(content, (str, list)):
                    raise ValueError(
                        f"Invalid content type: {type(content)}. "
                        "Content must be str or list of content blocks."
                    )

                # If content is a list, validate each block
                if isinstance(content, list):
                    for i, block in enumerate(content):
                        if not isinstance(block, dict):
                            raise ValueError(
                                f"Invalid content block at index {i}: {block}. "
                                "Each block must be a dict."
                            )
                        if "type" not in block:
                            raise ValueError(
                                f"Invalid content block at index {i}: {block}. "
                                "Each block must have a 'type' key."
                            )

                # Message is valid, add it with all original fields
                # Preserve special fields like tool_calls, tool_call_id, etc.
                normalized_msg: dict[str, Any] = {
                    "role": msg["role"],
                    "content": content,
                }

                # Preserve tool_calls field for assistant messages
                if "tool_calls" in msg:
                    normalized_msg["tool_calls"] = msg["tool_calls"]

                # Preserve tool_call_id field for tool messages
                if "tool_call_id" in msg:
                    normalized_msg["tool_call_id"] = msg["tool_call_id"]

                # Preserve any other fields (for extensibility)
                for key, value in msg.items():
                    if key not in ("role", "content", "tool_calls", "tool_call_id"):
                        normalized_msg[key] = value

                result.append(normalized_msg)
            else:
                raise ValueError(
                    f"Invalid message type: {type(msg)}. Expected str or dict."
                )
    else:
        raise ValueError(
            f"Invalid messages type: {type(messages)}. Expected str, list, or tuple."
        )

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
        output_tokens=usage_data.get("completion_tokens")
        or usage_data.get("output_tokens"),
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
