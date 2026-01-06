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

from lexilux import Chat, ChatContinue, ChatHistory, ChatResult
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
    def test_call_with_history_updates_history(self, mock_post):
        """Test that chat() updates history when provided"""
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
        result = chat("Hello", history=history)

        # History should be updated
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[1]["content"] == "Hello!"
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
        chat("Hello", history=history)
        chat("How are you?", history=history)

        # Check that second call included history
        assert mock_post.call_count == 2
        second_call_payload = mock_post.call_args_list[1].kwargs["json"]
        messages = second_call_payload["messages"]
        # Should have: Hello + Hi! (from history) + How are you? (new message)
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"
        assert messages[2]["content"] == "How are you?"

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
    def test_call_with_history_user_message_added_before_request(self, mock_post):
        """Test that user message is added to history before request (even if request fails)"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Network error")
        mock_post.return_value = mock_response

        history = ChatHistory()
        with pytest.raises(requests.RequestException):
            chat("Hello", history=history)

        # User message should be added even if request fails
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"
        # No assistant message (request failed)

    @patch("lexilux.chat.client.requests.post")
    def test_stream_with_history_updates_history(self, mock_post):
        """Test that stream() updates history when provided"""
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
        iterator = chat.stream("Hello", history=history)

        # Iterate all chunks
        chunks = list(iterator)

        # History should be updated
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[1]["content"] == "Hello world"

    @patch("lexilux.chat.client.requests.post")
    def test_stream_with_history_lazy_assistant_initialization(self, mock_post):
        """Test that assistant message is added only on first iteration"""
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
        iterator = chat.stream("Hello", history=history)

        # Before iteration: user message should be added, but no assistant
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"

        # First iteration: assistant message should be added
        iter_obj = iter(iterator)
        next(iter_obj)
        assert len(history.messages) == 2
        assert history.messages[1]["role"] == "assistant"


class TestChatContinue:
    """Test Chat.continue_if_needed() method (v2.0 - requires explicit history)"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_if_needed_continues_when_truncated(self, mock_post):
        """Test that continue_if_needed continues when finish_reason='length'"""
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

        # Continue with explicit history
        full_result = chat.continue_if_needed(result, history=history)

        assert full_result.finish_reason == "stop"
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text

    @patch("lexilux.chat.client.requests.post")
    def test_continue_if_needed_returns_unchanged_when_not_truncated(self, mock_post):
        """Test that continue_if_needed returns result unchanged when not truncated"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Complete response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        result = chat("Write a story", history=history, max_tokens=100)
        assert result.finish_reason == "stop"

        # Should return unchanged
        full_result = chat.continue_if_needed(result, history=history)

        assert full_result is result  # Same object
        assert full_result.finish_reason == "stop"
        assert full_result.text == "Complete response"

    def test_continue_if_needed_requires_history(self):
        """Test that continue_if_needed requires history parameter"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        # Should raise TypeError (missing required argument)
        with pytest.raises(TypeError):
            chat.continue_if_needed(result)


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
            chat.complete("Write a very long story", history=history, max_tokens=10, max_continues=2)

        assert exc_info.value.continue_count == 2
        assert exc_info.value.max_continues == 2
        assert exc_info.value.final_result.finish_reason == "length"

    def test_complete_requires_history(self):
        """Test that complete() requires history parameter"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Should raise TypeError (missing required argument)
        with pytest.raises(TypeError):
            chat.complete("Write a story")


class TestChatStreamingContinue:
    """Test streaming continue methods (v2.0)"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_if_needed_stream_continues_when_truncated(self, mock_post):
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

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        result = chat("Write a story", history=history, max_tokens=50)
        assert result.finish_reason == "length"

        # Stream continue
        iterator = chat.continue_if_needed_stream(result, history=history)
        chunks = list(iterator)

        # Should have chunks from continue
        assert len(chunks) > 0
        # Result should be merged
        full_result = iterator.result.to_chat_result()
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text

    @patch("lexilux.chat.client.requests.post")
    def test_continue_if_needed_stream_returns_done_chunk_when_not_truncated(self, mock_post):
        """Test that continue_if_needed_stream returns done chunk when not truncated"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Complete"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        result = chat("Write a story", history=history)
        assert result.finish_reason == "stop"

        # Stream continue (should return single done chunk)
        iterator = chat.continue_if_needed_stream(result, history=history)
        chunks = list(iterator)

        # Should have one done chunk
        assert len(chunks) == 1
        assert chunks[0].done is True
        assert chunks[0].delta == ""

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

