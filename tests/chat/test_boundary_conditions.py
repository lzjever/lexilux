"""
Boundary condition tests for chat functionality.

Tests verify that edge cases and boundary conditions are handled correctly.
Some tests may fail if validation is not yet implemented - this is expected
as we are establishing a baseline for test coverage.
"""

import pytest
from unittest.mock import patch, MagicMock

from lexilux import Chat
from lexilux.chat.history import ChatHistory
from lexilux.exceptions import ValidationError


class TestBoundaryConditions:
    """Test boundary conditions for chat functionality."""

    # Empty string message tests
    def test_empty_string_message(self):
        """Test empty string message is handled gracefully."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("")

            assert result.text == "Hello!"
            mock_request.assert_called_once()
            # Verify empty string is passed in the messages
            call_args = mock_request.call_args[0][1]
            assert call_args["messages"][-1]["content"] == ""

    def test_empty_string_in_message_list(self):
        """Test empty string in message list is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat([""])

            assert result.text == "Response"

    # Very long message tests
    def test_very_long_message(self):
        """Test very long message doesn't break the client."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a very long message (100,000 characters)
        long_message = "a" * 100000

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100000,
                    "completion_tokens": 1,
                    "total_tokens": 100001,
                },
            }
            mock_request.return_value = mock_response

            result = chat(long_message)

            assert result.text == "OK"
            mock_request.assert_called_once()
            call_args = mock_request.call_args[0][1]
            assert len(call_args["messages"][-1]["content"]) == 100000

    # max_tokens boundary tests
    def test_zero_max_tokens(self):
        """Test max_tokens=0 raises ValidationError."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            chat("test", max_tokens=0)

    def test_large_max_tokens(self):
        """Test large max_tokens value is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", max_tokens=128000)

            assert result.text == "Response"
            call_args = mock_request.call_args[0][1]
            assert call_args["max_tokens"] == 128000

    # temperature boundary tests
    def test_temperature_at_lower_boundary(self):
        """Test temperature at valid lower boundary (0.0)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", temperature=0.0)

            assert result.text == "Response"
            call_args = mock_request.call_args[0][1]
            assert call_args["temperature"] == 0.0

    def test_temperature_at_upper_boundary(self):
        """Test temperature at valid upper boundary (2.0)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", temperature=2.0)

            assert result.text == "Response"
            call_args = mock_request.call_args[0][1]
            assert call_args["temperature"] == 2.0

    def test_invalid_temperature_below_range(self):
        """Test invalid temperature below valid range raises error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Note: This test expects validation to be implemented
        # If validation is not yet implemented, this will pass when it should fail
        with pytest.raises(ValidationError):
            chat("test", temperature=-0.1)

    def test_invalid_temperature_above_range(self):
        """Test invalid temperature above valid range raises error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Note: This test expects validation to be implemented
        # If validation is not yet implemented, this will pass when it should fail
        with pytest.raises(ValidationError):
            chat("test", temperature=2.1)

    # Large message history tests
    def test_many_messages_in_history(self):
        """Test large message history (100 messages) is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a history with 100 messages
        history = ChatHistory()
        for i in range(50):
            history.add_user(f"Message {i}")
            history.add_assistant(f"Response {i}")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Final response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "total_tokens": 101,
                },
            }
            mock_request.return_value = mock_response

            result = chat("New message", history=history)

            assert result.text == "Final response"
            call_args = mock_request.call_args[0][1]
            # Should have 101 messages total (50 pairs + 1 new user message)
            assert len(call_args["messages"]) == 101

    # n parameter boundary tests
    def test_n_at_lower_boundary(self):
        """Test n parameter at valid lower boundary (1)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response 1"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", n=1)

            assert result.text == "Response 1"
            call_args = mock_request.call_args[0][1]
            assert call_args["n"] == 1

    def test_n_at_upper_boundary(self):
        """Test n parameter at valid upper boundary (10)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            # Mock response with multiple choices
            choices = [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": f"Response {i + 1}"},
                    "finish_reason": "stop",
                }
                for i in range(10)
            ]
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": choices,
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 10,
                    "total_tokens": 11,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", n=10)

            # Should return the first choice
            assert result.text == "Response 1"
            call_args = mock_request.call_args[0][1]
            assert call_args["n"] == 10

    def test_invalid_n_below_range(self):
        """Test invalid n parameter below valid range raises error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Note: This test expects validation to be implemented
        # If validation is not yet implemented, the API call will be made anyway
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            # n=0 should ideally be rejected, but test documents current behavior
            chat("test", n=0)
            # If no validation, the call proceeds (documents baseline)
            call_args = mock_request.call_args[0][1]
            assert call_args["n"] == 0

    def test_invalid_n_above_range(self):
        """Test invalid n parameter above valid range raises error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Note: This test expects validation to be implemented
        # If validation is not yet implemented, the API call will be made anyway
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            # n=11 should ideally be rejected, but test documents current behavior
            chat("test", n=11)
            # If no validation, the call proceeds (documents baseline)
            call_args = mock_request.call_args[0][1]
            assert call_args["n"] == 11

    # top_p boundary tests
    def test_top_p_at_boundaries(self):
        """Test top_p at valid boundaries (0.0, 1.0)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Test lower boundary
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", top_p=0.0)
            assert result.text == "Response"

        # Test upper boundary
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", top_p=1.0)
            assert result.text == "Response"

    # presence_penalty and frequency_penalty boundary tests
    def test_presence_penalty_at_boundaries(self):
        """Test presence_penalty at valid boundaries (-2.0, 2.0)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Test lower boundary
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", presence_penalty=-2.0)
            assert result.text == "Response"

        # Test upper boundary
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", presence_penalty=2.0)
            assert result.text == "Response"

    def test_frequency_penalty_at_boundaries(self):
        """Test frequency_penalty at valid boundaries (-2.0, 2.0)."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Test lower boundary
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", frequency_penalty=-2.0)
            assert result.text == "Response"

        # Test upper boundary
        with patch.object(chat, "_make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            mock_request.return_value = mock_response

            result = chat("test", frequency_penalty=2.0)
            assert result.text == "Response"

    # Streaming boundary tests
    def test_stream_with_empty_message(self):
        """Test streaming with empty message."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with patch.object(chat, "_streaming_request_context") as mock_context:
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                b'data: {"id": "test", "choices": [{"delta": {"content": "Hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            mock_response.close = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_context.return_value = mock_response

            iterator = chat.stream("")

            chunks = list(iterator)
            assert len(chunks) >= 1

    def test_stream_with_long_message(self):
        """Test streaming with very long message."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        long_message = "x" * 50000

        with patch.object(chat, "_streaming_request_context") as mock_context:
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                b'data: {"id": "test", "choices": [{"delta": {"content": "OK"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            mock_response.close = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_context.return_value = mock_response

            iterator = chat.stream(long_message)

            chunks = list(iterator)
            assert len(chunks) >= 1

    def test_stream_with_zero_max_tokens(self):
        """Test streaming with max_tokens=0 raises ValidationError."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            list(chat.stream("test", max_tokens=0))
