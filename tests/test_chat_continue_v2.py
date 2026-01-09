"""
Comprehensive tests for ChatContinue v2.0 (with streaming support).

Tests are written based on the public interface specification, not implementation details.
Tests challenge the business logic and verify correct behavior according to the API contract.

v2.0 API changes:
1. continue_request() requires explicit history parameter
2. Added continue_request_stream() method
"""

from unittest.mock import Mock, patch

import pytest

from lexilux import Chat, ChatContinue, ChatHistory, ChatResult
from lexilux.usage import Usage


class TestChatContinueContinueRequest:
    """Test ChatContinue.continue_request() method (v2.0 - explicit history)"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_with_explicit_history(self, mock_post):
        """Test continue_request with explicit history"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result1 = ChatResult(
            text="This is part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        history = ChatHistory.from_messages("Write a story")
        history.append_result(result1)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " and part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        continue_result = ChatContinue.continue_request(
            chat, result1, history=history, add_continue_prompt=True, continue_prompt="continue"
        )

        # Should return merged result (auto_merge=True by default)
        assert isinstance(continue_result, ChatResult)
        assert "This is part 1" in continue_result.text
        assert " and part 2" in continue_result.text
        assert continue_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_requires_history(self, mock_post):
        """Test that continue_request requires history parameter"""
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

        # Should raise ValueError when history=None
        with pytest.raises(ValueError, match="History is required"):
            ChatContinue.continue_request(chat, result, history=None)

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_requires_length_finish_reason(self, mock_post):
        """Test that continue_request requires finish_reason='length'"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result = ChatResult(
            text="Complete response",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="stop",  # Not "length"
        )

        history = ChatHistory()
        # Should raise ValueError
        with pytest.raises(ValueError, match="finish_reason='length'"):
            ChatContinue.continue_request(chat, result, history=history)

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_max_continues(self, mock_post):
        """Test that max_continues allows multiple continuation attempts"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial result
        result1 = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )

        history = ChatHistory.from_messages("Write a long story")
        history.append_result(result1)

        # First continue (still truncated)
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": " Part 2"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response2.raise_for_status = Mock()

        # Second continue (complete)
        mock_response3 = Mock()
        mock_response3.status_code = 200
        mock_response3.json.return_value = {
            "choices": [{"message": {"content": " Part 3"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
        mock_response3.raise_for_status = Mock()

        mock_post.side_effect = [mock_response2, mock_response3]

        # Continue with max_continues=2
        full_result = ChatContinue.continue_request(chat, result1, history=history, max_continues=2)

        # Should have made 2 continue calls
        assert mock_post.call_count == 2

        # Should be merged result
        assert isinstance(full_result, ChatResult)
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text
        assert "Part 3" in full_result.text
        assert full_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_auto_merge_false_returns_list(self, mock_post):
        """Test that auto_merge=False returns list of results"""
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

        # Continue call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " Part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Continue with auto_merge=False
        all_results = ChatContinue.continue_request(
            chat, result1, history=history, auto_merge=False
        )

        # Should return list
        assert isinstance(all_results, list)
        assert len(all_results) == 2  # result1 + continue_result
        assert all(isinstance(r, ChatResult) for r in all_results)
        assert all_results[0].text == "Part 1"
        assert all_results[1].text == " Part 2"


class TestChatContinueContinueRequestStream:
    """Test ChatContinue.continue_request_stream() method (v2.0)"""

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_basic(self, mock_post):
        """Test basic continue_request_stream functionality"""
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

        # Stream response
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

        # Should yield chunks
        chunks = list(iterator)
        assert len(chunks) > 0

        # Result should be merged
        full_result = iterator.result.to_chat_result()
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text
        assert full_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_requires_history(self, mock_post):
        """Test that continue_request_stream requires history parameter"""
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

        # Should raise ValueError when history=None
        with pytest.raises(ValueError, match="History is required"):
            ChatContinue.continue_request_stream(chat, result, history=None)

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_requires_length_finish_reason(self, mock_post):
        """Test that continue_request_stream requires finish_reason='length'"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        result = ChatResult(
            text="Complete",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="stop",  # Not "length"
        )

        history = ChatHistory()
        # Should raise ValueError
        with pytest.raises(ValueError, match="finish_reason='length'"):
            ChatContinue.continue_request_stream(chat, result, history=history)

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_max_continues(self, mock_post):
        """Test that continue_request_stream handles multiple continues"""
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

        # Should yield chunks from both continues
        chunks = list(iterator)
        assert len(chunks) > 0

        # Result should be merged from all continues
        full_result = iterator.result.to_chat_result()
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text
        assert "Part 3" in full_result.text
        assert full_result.finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_continue_request_stream_history_immutability(self, mock_post):
        """Test that continue_request_stream does not modify original history (immutable)"""
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

        initial_count = len(history.messages)
        initial_messages = history.messages.copy()

        # Stream response
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

        # Iterate all chunks
        list(iterator)

        # Original history should NOT be modified (immutable)
        # Working history is cloned internally
        assert len(history.messages) == initial_count
        assert history.messages == initial_messages

        # Verify result contains merged text
        final_result = iterator.result.to_chat_result()
        assert "Part 1" in final_result.text
        assert "Part 2" in final_result.text or " Part 2" in final_result.text


class TestChatContinueMergeResults:
    """Test ChatContinue.merge_results() method"""

    def test_merge_results_single_result(self):
        """Test merge_results with single result"""
        result = ChatResult(
            text="Hello",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

        merged = ChatContinue.merge_results(result)
        assert merged is result  # Should return same object for single result

    def test_merge_results_multiple_results(self):
        """Test merge_results with multiple results"""
        result1 = ChatResult(
            text="Part 1",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )
        result2 = ChatResult(
            text=" Part 2",
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            finish_reason="stop",
        )

        merged = ChatContinue.merge_results(result1, result2)

        # Should merge text
        assert merged.text == "Part 1 Part 2"

        # Should merge usage
        assert merged.usage.input_tokens == 20
        assert merged.usage.output_tokens == 70
        assert merged.usage.total_tokens == 90

        # Should use last result's finish_reason
        assert merged.finish_reason == "stop"

    def test_merge_results_empty_raises(self):
        """Test that merge_results raises when no results provided"""
        with pytest.raises(ValueError, match="At least one result is required"):
            ChatContinue.merge_results()
