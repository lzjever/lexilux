"""
Tests for chat validation functions.

Tests verify that validation functions correctly validate input parameters.
"""

import pytest

from lexilux.chat.validation import (
    validate_chat_params,
    validate_messages,
    validate_model,
    validate_stop,
)
from lexilux.exceptions import ValidationError


class TestValidateChatParams:
    """Test validate_chat_params function."""

    # Temperature tests
    def test_valid_temperature_range(self):
        """Test valid temperature values within range."""
        # Should not raise for valid values
        validate_chat_params(
            temperature=0.0,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=1.0,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=2.0,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )

    def test_invalid_temperature_below_zero(self):
        """Test temperature below 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=-0.1,
                top_p=None,
                max_tokens=None,
                presence_penalty=None,
                frequency_penalty=None,
            )
        assert "temperature" in str(exc_info.value).lower()

    def test_invalid_temperature_above_two(self):
        """Test temperature above 2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=2.1,
                top_p=None,
                max_tokens=None,
                presence_penalty=None,
                frequency_penalty=None,
            )
        assert "temperature" in str(exc_info.value).lower()

    # Top_p tests
    def test_valid_top_p_range(self):
        """Test valid top_p values within range."""
        validate_chat_params(
            temperature=None,
            top_p=0.0,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=None,
            top_p=0.5,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=None,
            top_p=1.0,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )

    def test_invalid_top_p_below_zero(self):
        """Test top_p below 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=-0.1,
                max_tokens=None,
                presence_penalty=None,
                frequency_penalty=None,
            )
        assert "top_p" in str(exc_info.value).lower()

    def test_invalid_top_p_above_one(self):
        """Test top_p above 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=1.1,
                max_tokens=None,
                presence_penalty=None,
                frequency_penalty=None,
            )
        assert "top_p" in str(exc_info.value).lower()

    # Max tokens tests
    def test_valid_max_tokens(self):
        """Test valid max_tokens values."""
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=1,
            presence_penalty=None,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=1000,
            presence_penalty=None,
            frequency_penalty=None,
        )

    def test_invalid_max_tokens_zero(self):
        """Test max_tokens=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=None,
                max_tokens=0,
                presence_penalty=None,
                frequency_penalty=None,
            )
        assert "max_tokens" in str(exc_info.value).lower()

    def test_invalid_max_tokens_negative(self):
        """Test negative max_tokens raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=None,
                max_tokens=-1,
                presence_penalty=None,
                frequency_penalty=None,
            )
        assert "max_tokens" in str(exc_info.value).lower()

    # Presence penalty tests
    def test_valid_presence_penalty(self):
        """Test valid presence_penalty values within range."""
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=-2.0,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=0.0,
            frequency_penalty=None,
        )
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=2.0,
            frequency_penalty=None,
        )

    def test_invalid_presence_penalty_below_range(self):
        """Test presence_penalty below -2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=None,
                max_tokens=None,
                presence_penalty=-2.1,
                frequency_penalty=None,
            )
        assert "presence_penalty" in str(exc_info.value).lower()

    def test_invalid_presence_penalty_above_range(self):
        """Test presence_penalty above 2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=None,
                max_tokens=None,
                presence_penalty=2.1,
                frequency_penalty=None,
            )
        assert "presence_penalty" in str(exc_info.value).lower()

    # Frequency penalty tests
    def test_valid_frequency_penalty(self):
        """Test valid frequency_penalty values within range."""
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=-2.0,
        )
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=0.0,
        )
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=2.0,
        )

    def test_invalid_frequency_penalty_below_range(self):
        """Test frequency_penalty below -2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=None,
                max_tokens=None,
                presence_penalty=None,
                frequency_penalty=-2.1,
            )
        assert "frequency_penalty" in str(exc_info.value).lower()

    def test_invalid_frequency_penalty_above_range(self):
        """Test frequency_penalty above 2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_chat_params(
                temperature=None,
                top_p=None,
                max_tokens=None,
                presence_penalty=None,
                frequency_penalty=2.1,
            )
        assert "frequency_penalty" in str(exc_info.value).lower()

    # None values should be allowed
    def test_all_none_values_allowed(self):
        """Test that None values for all params is valid."""
        validate_chat_params(
            temperature=None,
            top_p=None,
            max_tokens=None,
            presence_penalty=None,
            frequency_penalty=None,
        )


class TestValidateMessages:
    """Test validate_messages function."""

    def test_valid_string_message(self):
        """Test valid string message."""
        result = validate_messages("Hello")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_valid_list_message(self):
        """Test valid list of message dicts."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = validate_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_valid_list_of_strings(self):
        """Test valid list of strings."""
        result = validate_messages(["Hello", "World"])
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "World"

    def test_invalid_empty_message(self):
        """Test empty message list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages([])
        assert "empty" in str(exc_info.value).lower()

    def test_invalid_message_type(self):
        """Test invalid message type raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_messages(123)  # type: ignore

    def test_valid_tool_message_with_content(self):
        """Test tool message with content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "123", "content": "tool response"},
        ]
        result = validate_messages(messages)
        assert len(result) == 2
        assert result[1]["role"] == "tool"

    def test_invalid_role(self):
        """Test invalid role raises ValidationError."""
        messages = [{"role": "invalid_role", "content": "Hello"}]
        with pytest.raises(ValidationError) as exc_info:
            validate_messages(messages)
        assert "role" in str(exc_info.value).lower()


class TestValidateModel:
    """Test validate_model function."""

    def test_valid_model(self):
        """Test valid model name."""
        result = validate_model("gpt-4", None)
        assert result == "gpt-4"

    def test_model_from_default(self):
        """Test model from default when not specified."""
        result = validate_model(None, "gpt-3.5-turbo")
        assert result == "gpt-3.5-turbo"

    def test_invalid_empty_model(self):
        """Test empty model raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model(None, None)
        assert "model" in str(exc_info.value).lower()

    def test_invalid_whitespace_model(self):
        """Test whitespace-only model raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_model("   ", None)


class TestValidateStop:
    """Test validate_stop function."""

    def test_valid_stop_string(self):
        """Test valid stop string."""
        result = validate_stop("stop")
        assert result == ["stop"]

    def test_valid_stop_list(self):
        """Test valid stop list."""
        result = validate_stop(["stop1", "stop2"])
        assert result == ["stop1", "stop2"]

    def test_none_stop(self):
        """Test None stop returns None."""
        result = validate_stop(None)
        assert result is None

    def test_invalid_stop_too_many(self):
        """Test stop list with too many items (API typically limits to 4)."""
        # This test documents current behavior - may not enforce limit
        result = validate_stop(["a", "b", "c", "d", "e"])
        assert len(result) == 5

    def test_invalid_empty_stop_string(self):
        """Test empty stop string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_stop("")

    def test_invalid_empty_stop_list(self):
        """Test empty stop list raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_stop([])

    def test_invalid_stop_type(self):
        """Test invalid stop type raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_stop(123)  # type: ignore
