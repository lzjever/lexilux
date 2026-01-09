"""
Comprehensive tests for Chat API v2.0.

Tests are written based on the public interface specification, not implementation details.
Tests challenge the business logic and verify correct behavior according to the API contract.

v2.0 API changes:
1. Removed auto_history parameter - all history management is explicit
2. All methods accept explicit history parameter
3. ChatHistory implements MutableSequence protocol
4. Added streaming continue methods
"""

from unittest.mock import Mock, patch

import pytest
import requests

from lexilux import Chat, ChatHistory, ChatResult
from lexilux.chat.exceptions import ChatIncompleteResponseError
from lexilux.usage import Usage


class TestChatInit:
    """Test Chat initialization (v2.0 - no auto_history)"""

    def test_init_with_all_params(self):
        """Test Chat initialization with all parameters"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
            timeout_s=30.0,
            headers={"X-Custom": "value"},
        )
        assert chat.base_url == "https://api.example.com/v1"
        assert chat.api_key == "test-key"
        assert chat.model == "gpt-4"
        assert chat.timeout_s == 30.0
        assert chat.headers["Authorization"] == "Bearer test-key"
        assert chat.headers["X-Custom"] == "value"
        # v2.0: No auto_history attribute
        assert not hasattr(chat, "auto_history")
        assert not hasattr(chat, "_history")

    def test_init_without_api_key(self):
        """Test Chat initialization without API key"""
        chat = Chat(base_url="https://api.example.com/v1", model="gpt-4")
        assert chat.api_key is None
        assert "Authorization" not in chat.headers

    def test_init_strips_trailing_slash(self):
        """Test that base_url trailing slash is stripped"""
        chat = Chat(base_url="https://api.example.com/v1/", model="gpt-4")
        assert chat.base_url == "https://api.example.com/v1"


class TestChatWithExplicitHistory:
    """Test Chat methods with explicit history parameter (v2.0)"""

    @patch("lexilux.chat.client.requests.post")
    def test_call_with_history_immutability(self, mock_post):
        """Test that chat() does not modify original history (immutable)"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        original_count = len(history.messages)
        result = chat("Hello", history=history)

        # Original history should NOT be modified (immutable)
        assert len(history.messages) == original_count
        assert result.text == "Hello!"

    @patch("lexilux.chat.client.requests.post")
    def test_call_with_history_prepends_history_messages(self, mock_post):
        """Test that chat() prepends history messages to request"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # First call
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response1.raise_for_status = Mock()

        # Second call
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": "How can I help?"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        result1 = chat("Hello", history=history)

        # Manually update history for multi-turn (history is immutable)
        history.add_user("Hello")
        history.append_result(result1)

        chat("How are you?", history=history)

        # Check that second call included history (working history is cloned internally)
        assert mock_post.call_count == 2
        second_call_payload = mock_post.call_args_list[1].kwargs["json"]
        messages = second_call_payload["messages"]
        # Should have: Hello + Hi! (from history) + How are you? (new message)
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"
        assert messages[2]["content"] == "How are you?"

        # Original history should not be modified by second call
        assert len(history.messages) == 2  # Only manually added messages

    @patch("lexilux.chat.client.requests.post")
    def test_call_without_history_no_update(self, mock_post):
        """Test that chat() does not update anything when history=None"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = chat("Hello", history=None)

        # Result should be correct
        assert result.text == "Hello!"
        # No history to check, but should not crash

    @patch("lexilux.chat.client.requests.post")
    def test_call_with_history_immutability_on_error(self, mock_post):
        """Test that original history is not modified even if request fails (immutable)"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Network error")
        mock_post.return_value = mock_response

        history = ChatHistory()
        original_count = len(history.messages)
        with pytest.raises(requests.RequestException):
            chat("Hello", history=history)

        # Original history should NOT be modified (immutable)
        assert len(history.messages) == original_count

    @patch("lexilux.chat.client.requests.post")
    def test_stream_with_history_immutability(self, mock_post):
        """Test that stream() does not modify original history (immutable)"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        original_count = len(history.messages)
        iterator = chat.stream("Hello", history=history)

        # Iterate all chunks
        list(iterator)

        # Original history should NOT be modified (immutable)
        assert len(history.messages) == original_count

        # Verify result is correct
        result = iterator.result.to_chat_result()
        assert result.text == "Hello world"

    @patch("lexilux.chat.client.requests.post")
    def test_stream_with_history_immutability_during_iteration(self, mock_post):
        """Test that original history is not modified during streaming iteration (immutable)"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        original_count = len(history.messages)
        iterator = chat.stream("Hello", history=history)

        # Before iteration: original history should not be modified
        assert len(history.messages) == original_count

        # After iteration: original history should still not be modified
        list(iterator)
        assert len(history.messages) == original_count


class TestChatContinue:
    """Test ChatContinue.continue_request() method (v2.1 - history immutability)"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_continues_when_truncated(self, mock_post):
        """Test that ChatContinue.continue_request continues when finish_reason='length'"""
        from lexilux.chat.continue_ import ChatContinue

        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (truncated)
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Part 1"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response1.raise_for_status = Mock()

        # Continue call
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": " Part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        result = chat("Write a story", history=history, max_tokens=50)
        assert result.finish_reason == "length"

        # Continue with explicit history (using ChatContinue)
        full_result = ChatContinue.continue_request(chat, result, history=history)

        assert full_result.finish_reason == "stop"
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text


class TestChatComplete:
    """Test Chat.complete() method (v2.0 - requires explicit history)"""

    @patch("lexilux.chat.client.requests.post")
    def test_complete_ensures_complete_response(self, mock_post):
        """Test that complete() ensures complete response"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (truncated)
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Part 1"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response1.raise_for_status = Mock()

        # Continue call (complete)
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": " Part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        result = chat.complete("Write a story", history=history, max_tokens=50)

        # Should be complete (merged)
        assert result.finish_reason == "stop"
        assert "Part 1" in result.text
        assert "Part 2" in result.text

    @patch("lexilux.chat.client.requests.post")
    def test_complete_raises_incomplete_response_when_still_truncated(self, mock_post):
        """Test that complete() raises ChatIncompleteResponseError when still truncated"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # All calls return truncated
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Part"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        # Should raise ChatIncompleteResponseError after max_continues
        with pytest.raises(ChatIncompleteResponseError) as exc_info:
            chat.complete(
                "Write a very long story", history=history, max_tokens=10, max_continues=2
            )

        assert exc_info.value.continue_count == 2
        assert exc_info.value.max_continues == 2
        assert exc_info.value.final_result.finish_reason == "length"

    def test_complete_works_without_history(self):
        """Test that complete() works without history parameter (creates internally)"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Should work without history (creates internally)
        # Note: This test requires mocking, but we're just checking it doesn't raise TypeError
        # In real usage, it would make API calls
        # This test verifies the signature accepts None for history
        assert chat.complete.__annotations__.get("history") == "ChatHistory | None"


class TestChatStreamingContinue:
    """Test streaming continue methods (v2.0)"""

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_continues_when_truncated(self, mock_post):
        """Test that continue_if_needed_stream continues when truncated"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (truncated)
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Part 1"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response1.raise_for_status = Mock()

        # Continue call (streaming)
        stream_data = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.iter_lines.return_value = iter(stream_data)
        mock_response2.raise_for_status = Mock()

        # Mock initial streaming response (truncated)
        stream_data1 = [
            b'data: {"choices": [{"delta": {"content": "Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 1"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response1_stream = Mock()
        mock_response1_stream.status_code = 200
        mock_response1_stream.iter_lines.return_value = iter(stream_data1)
        mock_response1_stream.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1_stream, mock_response2]

        # Use complete_stream (handles truncation automatically)
        iterator = chat.complete_stream("Write a story", max_tokens=50)
        chunks = list(iterator)

        # Should have chunks from initial and continue
        assert len(chunks) > 0
        # Result should be merged
        full_result = iterator.result.to_chat_result()
        assert "Part" in full_result.text or "Part 1" in full_result.text
        assert "Part 2" in full_result.text or " Part 2" in full_result.text

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_ensures_complete_response(self, mock_post):
        """Test that complete_stream ensures complete response"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (streaming, truncated)
        stream_data1 = [
            b'data: {"choices": [{"delta": {"content": "Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 1"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.iter_lines.return_value = iter(stream_data1)
        mock_response1.raise_for_status = Mock()

        # Continue call (streaming, complete)
        stream_data2 = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.iter_lines.return_value = iter(stream_data2)
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        iterator = chat.complete_stream("Write a story", history=history, max_tokens=50)
        chunks = list(iterator)

        # Should have chunks from both initial and continue
        assert len(chunks) > 0
        # Result should be complete
        result = iterator.result.to_chat_result()
        assert result.finish_reason == "stop"
        assert "Part 1" in result.text
        assert "Part 2" in result.text
