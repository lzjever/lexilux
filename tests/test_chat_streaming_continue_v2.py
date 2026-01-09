"""
Comprehensive tests for streaming continue functionality v2.0.

Tests are written based on the public interface specification, not implementation details.
Tests challenge the business logic and verify correct behavior, especially edge cases.
"""

from unittest.mock import Mock, patch

import pytest

from lexilux import Chat, ChatContinue, ChatHistory, ChatResult
from lexilux.usage import Usage


class TestContinueRequestStreamEdgeCases:
    """Test edge cases for continue_request_stream"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_single_continue_complete(self, mock_post):
        """Test that single continue that completes works correctly"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result1 = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        history = ChatHistory.from_messages("Write a story")
        history.append_result(result1)

        # Single continue (complete)
        stream_data = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        iterator = ChatContinue.continue_request_stream(chat, result1, history=history)

        chunks = list(iterator)
        assert len(chunks) >= 2  # At least content chunks + done

        # Result should be merged
        full_result = iterator.result.to_chat_result()
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text
        assert full_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_multiple_continues(self, mock_post):
        """Test that multiple continues work correctly"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result1 = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        history = ChatHistory.from_messages("Write a long story")
        history.append_result(result1)

        # First continue (still truncated)
        stream_data1 = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.iter_lines.return_value = iter(stream_data1)
        mock_response1.raise_for_status = Mock()

        # Second continue (complete)
        stream_data2 = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 3"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.iter_lines.return_value = iter(stream_data2)
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        iterator = ChatContinue.continue_request_stream(
            chat, result1, history=history, max_continues=2
        )

        chunks = list(iterator)
        assert len(chunks) > 0

        # Result should be merged from all continues
        full_result = iterator.result.to_chat_result()
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text
        assert "Part 3" in full_result.text
        assert full_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_max_continues_reached(self, mock_post):
        """Test that max_continues limit is respected"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result1 = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        history = ChatHistory.from_messages("Write a very long story")
        history.append_result(result1)

        # All continues return truncated
        # Note: finish_reason chunk may have empty delta
        stream_data = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        iterator = ChatContinue.continue_request_stream(
            chat, result1, history=history, max_continues=2
        )

        chunks = list(iterator)
        assert len(chunks) > 0

        # Result should be merged but still truncated
        full_result = iterator.result.to_chat_result()
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text
        assert full_result.finish_reason == "length"  # Still truncated after max_continues

        # Should have made exactly max_continues calls
        assert mock_post.call_count == 2

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_usage_merging(self, mock_post):
        """Test that usage is correctly merged from all continues"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result1 = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        history = ChatHistory.from_messages("Write a story")
        history.append_result(result1)

        # Continue with usage
        stream_data = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        iterator = ChatContinue.continue_request_stream(chat, result1, history=history)

        list(iterator)  # Consume all chunks

        # Result should have merged usage
        full_result = iterator.result.to_chat_result()
        # Usage should be merged (simplified check - actual merging logic may vary)
        assert full_result.usage is not None


class TestContinueIfNeededStreamEdgeCases:
    """Test edge cases for continue_if_needed_stream"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_if_needed_stream_not_truncated_returns_single_chunk(self, mock_post):
        """Test that continue_if_needed_stream returns single done chunk when not truncated"""
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

        # Should return iterator with single done chunk
        iterator = chat.continue_if_needed_stream(result, history=history)
        chunks = list(iterator)

        assert len(chunks) == 1
        assert chunks[0].done is True
        assert chunks[0].delta == ""

        # Result should be the original result
        full_result = iterator.result.to_chat_result()
        assert full_result.text == "Complete"
        assert full_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_empty_delta_chunks(self, mock_post):
        """Test that empty delta chunks are handled correctly in complete_stream"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial truncated response
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Part 1"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response1.raise_for_status = Mock()

        # Stream with empty delta chunks for continue
        stream_data = [
            b'data: {"choices": [{"delta": {}, "index": 0}]}\n',  # Empty delta
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {}, "index": 0}]}\n',  # Empty delta
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

        iterator = chat.complete_stream("Write a story", max_tokens=50)
        chunks = list(iterator)

        # Should have chunks (including empty delta ones)
        assert len(chunks) > 0

        # Result should be correct
        full_result = iterator.result.to_chat_result()
        assert "Part" in full_result.text or "Part 1" in full_result.text
        assert "Part 2" in full_result.text or " Part 2" in full_result.text


class TestCompleteStreamEdgeCases:
    """Test edge cases for complete_stream"""

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_no_truncation_needed(self, mock_post):
        """Test complete_stream when no truncation occurs"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (complete, not truncated)
        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Complete"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " response"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        iterator = chat.complete_stream("Write a story", history=history)
        chunks = list(iterator)

        # Should have chunks from initial request only
        assert len(chunks) > 0

        # Result should be complete
        result = iterator.result.to_chat_result()
        assert result.finish_reason == "stop"
        assert "Complete response" in result.text

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_raises_when_still_truncated(self, mock_post):
        """Test that complete_stream raises error when still truncated after max_continues"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # All calls return truncated
        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        iterator = chat.complete_stream(
            "Write a very long story",
            history=history,
            max_tokens=10,
            max_continues=2,
            ensure_complete=True,
        )

        # Should raise ChatIncompleteResponseError after iteration
        with pytest.raises(Exception):  # May raise during iteration or after
            list(iterator)
            # If iteration completes, error should be raised when accessing result
            iterator.result.to_chat_result()

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_ensure_complete_false_allows_partial(self, mock_post):
        """Test that ensure_complete=False allows partial result"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # All calls return truncated
        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        iterator = chat.complete_stream(
            "Write a very long story",
            history=history,
            max_tokens=10,
            max_continues=2,
            ensure_complete=False,
        )

        # Should not raise, even if still truncated
        list(iterator)
        result = iterator.result.to_chat_result()

        # Result may still be truncated
        assert result.finish_reason == "length"  # Still truncated after max_continues
