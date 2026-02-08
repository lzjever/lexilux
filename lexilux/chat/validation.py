"""
Input validation for chat client.

Provides validation functions for chat parameters, messages, and model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lexilux.chat.models import MessagesLike
from lexilux.exceptions import ValidationError


def validate_chat_params(
    *,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
) -> None:
    """
    Validate chat completion parameters.

    Args:
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling parameter (0-1).
        max_tokens: Maximum tokens to generate.
        presence_penalty: Presence penalty (-2 to 2).
        frequency_penalty: Frequency penalty (-2 to 2).

    Raises:
        ValidationError: If any parameter is invalid.
    """
    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise ValidationError(
                f"temperature must be a number, got {type(temperature).__name__}"
            )
        if not (0 <= temperature <= 2):
            raise ValidationError(
                f"temperature must be between 0 and 2, got {temperature}"
            )

    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            raise ValidationError(f"top_p must be a number, got {type(top_p).__name__}")
        if not (0 <= top_p <= 1):
            raise ValidationError(f"top_p must be between 0 and 1, got {top_p}")

    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            raise ValidationError(
                f"max_tokens must be an integer, got {type(max_tokens).__name__}"
            )
        if max_tokens < 1:
            raise ValidationError(f"max_tokens must be positive, got {max_tokens}")

    if presence_penalty is not None:
        if not isinstance(presence_penalty, (int, float)):
            raise ValidationError(
                f"presence_penalty must be a number, got {type(presence_penalty).__name__}"
            )
        if not (-2 <= presence_penalty <= 2):
            raise ValidationError(
                f"presence_penalty must be between -2 and 2, got {presence_penalty}"
            )

    if frequency_penalty is not None:
        if not isinstance(frequency_penalty, (int, float)):
            raise ValidationError(
                f"frequency_penalty must be a number, got {type(frequency_penalty).__name__}"
            )
        if not (-2 <= frequency_penalty <= 2):
            raise ValidationError(
                f"frequency_penalty must be between -2 and 2, got {frequency_penalty}"
            )


def validate_messages(messages: MessagesLike) -> list[dict[str, Any]]:
    """
    Validate and normalize messages.

    Args:
        messages: Messages to validate (string, dict, or list).

    Returns:
        Normalized messages as list of dicts.

    Raises:
        ValidationError: If messages are invalid.
    """
    from lexilux.chat.utils import normalize_messages

    try:
        normalized = normalize_messages(messages)
    except Exception as e:
        raise ValidationError(f"Invalid messages format: {e}") from e

    if not normalized:
        raise ValidationError("Messages cannot be empty")

    # Validate each message has required fields
    for msg in normalized:
        if not isinstance(msg, dict):
            raise ValidationError(
                f"Each message must be a dict, got {type(msg).__name__}"
            )

        if "role" not in msg:
            raise ValidationError(f"Message missing 'role' field: {msg}")

        if "content" not in msg and msg.get("role") != "tool":
            # Tool messages may not have content
            raise ValidationError(f"Message missing 'content' field: {msg}")

        valid_roles = {"system", "user", "assistant", "tool"}
        if msg["role"] not in valid_roles:
            raise ValidationError(
                f"Invalid role '{msg.get('role')}', must be one of {valid_roles}"
            )

    return normalized


def validate_model(
    model: str | None,
    default_model: str | None,
) -> str:
    """
    Validate model parameter.

    Args:
        model: Model specified in call.
        default_model: Default model from client initialization.

    Returns:
        Validated model name.

    Raises:
        ValidationError: If model is not specified.
    """
    final_model = model or default_model

    if not final_model:
        raise ValidationError(
            "Model must be specified (either in client initialization or in call)"
        )

    if not isinstance(final_model, str):
        raise ValidationError(
            f"Model must be a string, got {type(final_model).__name__}"
        )

    if not final_model.strip():
        raise ValidationError("Model cannot be empty or whitespace")

    return final_model


def validate_stop(stop: str | Sequence[str] | None) -> list[str] | None:
    """
    Validate stop sequences.

    Args:
        stop: Stop sequence(s).

    Returns:
        Normalized stop sequences as list, or None.

    Raises:
        ValidationError: If stop sequences are invalid.
    """
    if stop is None:
        return None

    if isinstance(stop, str):
        if not stop:
            raise ValidationError("Stop sequence cannot be empty string")
        return [stop]

    if isinstance(stop, Sequence):
        stop_list = list(stop)
        if not stop_list:
            raise ValidationError("Stop sequences cannot be empty list")
        for seq in stop_list:
            if not isinstance(seq, str):
                raise ValidationError(
                    f"Each stop sequence must be a string, got {type(seq).__name__}"
                )
            if not seq:
                raise ValidationError("Stop sequence cannot be empty string")
        return stop_list

    raise ValidationError(
        f"Stop must be a string or sequence of strings, got {type(stop).__name__}"
    )
