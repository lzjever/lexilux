"""
Tests for Chat API improvements:
1. History Immutability
2. Customizable continue strategy
3. chat() vs chat.complete() distinction
4. Removed continue_if_needed() methods
"""

import json
from unittest.mock import Mock, patch

import pytest

from lexilux import Chat, ChatHistory, ChatContinue
from lexilux.chat.exceptions import ChatIncompleteResponseError
from lexilux.chat.models import ChatResult
from lexilux.usage import Usage


class TestHistoryImmutability:
    """Test that history parameters are immutable."""

    def test_chat_call_history_immutability(self, mock_post):
        """Test that chat() does not modify original history."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Create original history
        original_history = ChatHistory()
        original_history.add_user("Previous message")
        original_messages_count = len(original_history.messages)

        # Mock response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Call chat with history
        result = chat("New message", history=original_history)

        # Verify original history is not modified
        assert len(original_history.messages) == original_messages_count
        assert original_history.messages == [{"role": "user", "content": "Previous message"}]

        # Verify result is correct
        assert result.text == "Hello"
        assert result.finish_reason == "stop"

    def test_chat_stream_history_immutability(self, mock_post):
        """Test that chat.stream() does not modify original history."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Create original history
        original_history = ChatHistory()
        original_history.add_user("Previous message")
        original_messages_count = len(original_history.messages)

        # Mock streaming response
        mock_post.return_value.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            b'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}',
            b"data: [DONE]",
        ]

        # Call stream with history
        iterator = chat.stream("New message", history=original_history)
        chunks = list(iterator)

        # Verify original history is not modified
        assert len(original_history.messages) == original_messages_count
        assert original_history.messages == [{"role": "user", "content": "Previous message"}]

        # Verify chunks
        assert len(chunks) >= 2

    def test_complete_history_immutability(self, mock_post):
        """Test that chat.complete() does not modify original history."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Create original history
        original_history = ChatHistory()
        original_history.add_user("Previous message")
        original_messages_count = len(original_history.messages)

        # Mock responses (initial + continue)
        responses = [
            {
                "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            },
        ]
        mock_post.side_effect = [
            Mock(json=lambda: responses[0], raise_for_status=lambda: None),
            Mock(json=lambda: responses[1], raise_for_status=lambda: None),
        ]

        # Call complete with history
        result = chat.complete("New message", history=original_history, max_tokens=10)

        # Verify original history is not modified
        assert len(original_history.messages) == original_messages_count
        assert original_history.messages == [{"role": "user", "content": "Previous message"}]

        # Verify result is merged
        assert "Partial" in result.text
        assert "complete" in result.text


class TestChatVsComplete:
    """Test distinction between chat() and chat.complete()."""

    def test_chat_accepts_truncation(self, mock_post):
        """Test that chat() accepts truncated responses."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Mock truncated response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Partial response"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        result = chat("Write a long story", max_tokens=10)

        # chat() should return truncated result without error
        assert result.finish_reason == "length"
        assert result.text == "Partial response"

    def test_complete_handles_truncation(self, mock_post):
        """Test that chat.complete() automatically handles truncation."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Mock responses (initial truncated + continue complete)
        responses = [
            {
                "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            },
        ]
        mock_post.side_effect = [
            Mock(json=lambda: responses[0], raise_for_status=lambda: None),
            Mock(json=lambda: responses[1], raise_for_status=lambda: None),
        ]

        result = chat.complete("Write a long story", max_tokens=10)

        # complete() should return complete result
        assert result.finish_reason == "stop"
        assert "Partial" in result.text
        assert "complete" in result.text

    def test_complete_without_history(self, mock_post):
        """Test that chat.complete() works without explicit history."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Mock complete response (no truncation)
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Complete response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Should work without history parameter
        result = chat.complete("Write a story", max_tokens=100)

        assert result.finish_reason == "stop"
        assert result.text == "Complete response"

    def test_complete_raises_if_still_truncated(self, mock_post):
        """Test that chat.complete() raises error if still truncated after max_continues."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Mock responses (all truncated)
        truncated_response = {
            "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_post.return_value.json.return_value = truncated_response
        mock_post.return_value.raise_for_status = lambda: None

        # Should raise error if still truncated
        with pytest.raises(ChatIncompleteResponseError):
            chat.complete(
                "Write a very long story", max_tokens=5, max_continues=2, ensure_complete=True
            )


class TestCustomizableContinueStrategy:
    """Test customizable continue strategy features."""

    def test_custom_continue_prompt_function(self, mock_post):
        """Test that continue_prompt can be a function."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")
        history = ChatHistory()

        # Mock responses
        responses = [
            {
                "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            },
        ]
        mock_post.side_effect = [
            Mock(json=lambda: responses[0], raise_for_status=lambda: None),
            Mock(json=lambda: responses[1], raise_for_status=lambda: None),
        ]

        # Custom prompt function
        prompt_calls = []

        def custom_prompt(count, max_count, current_text, original_prompt):
            prompt_calls.append((count, max_count, current_text, original_prompt))
            return f"Continue {count}/{max_count}"

        result = chat.complete(
            "Write story",
            history=history,
            max_tokens=10,
            continue_prompt=custom_prompt,
        )

        # Verify custom prompt was called
        assert len(prompt_calls) > 0
        assert prompt_calls[0][0] == 1  # First continue
        assert "Write story" in prompt_calls[0][3]  # Original prompt

    def test_progress_callback(self, mock_post):
        """Test that on_progress callback is called."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")
        history = ChatHistory()

        # Mock responses
        responses = [
            {
                "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            },
        ]
        mock_post.side_effect = [
            Mock(json=lambda: responses[0], raise_for_status=lambda: None),
            Mock(json=lambda: responses[1], raise_for_status=lambda: None),
        ]

        # Progress callback
        progress_calls = []

        def on_progress(count, max_count, current, all_results):
            progress_calls.append((count, max_count, len(current.text), len(all_results)))

        result = chat.complete(
            "Write story",
            history=history,
            max_tokens=10,
            on_progress=on_progress,
        )

        # Verify progress callback was called
        assert len(progress_calls) > 0
        assert progress_calls[0][0] == 1  # First continue
        assert progress_calls[0][1] >= 1  # max_count

    def test_continue_delay(self, mock_post):
        """Test that continue_delay is applied."""
        import time
        from unittest.mock import call

        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")
        history = ChatHistory()

        # Track call times
        call_times = []

        def track_time(*args, **kwargs):
            call_times.append(time.time())
            # Return appropriate response based on call count
            if len(call_times) == 1:
                # Initial request - truncated
                return Mock(
                    json=lambda: {
                        "choices": [
                            {"message": {"content": "Partial1"}, "finish_reason": "length"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                    raise_for_status=lambda: None,
                )
            elif len(call_times) == 2:
                # First continue - truncated
                return Mock(
                    json=lambda: {
                        "choices": [
                            {"message": {"content": "Partial2"}, "finish_reason": "length"}
                        ],
                        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
                    },
                    raise_for_status=lambda: None,
                )
            else:
                # Second continue - complete
                return Mock(
                    json=lambda: {
                        "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
                    },
                    raise_for_status=lambda: None,
                )

        mock_post.side_effect = track_time

        # Test with fixed delay
        result = chat.complete(
            "Write story",
            history=history,
            max_tokens=10,
            max_continues=3,
            continue_delay=0.05,  # 50ms delay
        )

        # Verify delay was applied between continues (call 2 and 3 should have delay)
        if len(call_times) >= 3:
            delay_between_continues = call_times[2] - call_times[1]
            assert delay_between_continues >= 0.05, (
                f"Expected delay >= 0.05s, got {delay_between_continues}"
            )

        assert result.finish_reason == "stop"

    def test_error_handling_return_partial(self, mock_post):
        """Test that on_error='return_partial' returns partial result on error."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")
        history = ChatHistory()

        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Initial request - truncated
                return Mock(
                    json=lambda: {
                        "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                    raise_for_status=lambda: None,
                )
            else:
                # Continue request fails
                raise Exception("Network error")

        mock_post.side_effect = mock_response

        # Should return partial result instead of raising
        result = chat.complete(
            "Write story",
            history=history,
            max_tokens=10,
            max_continues=1,
            ensure_complete=False,  # Don't raise if still truncated
            on_error="return_partial",
        )

        # Should return partial result (may still be truncated if error occurred)
        assert "Partial" in result.text


class TestContinueRequestCustomization:
    """Test ChatContinue.continue_request() customization features."""

    def test_continue_request_with_custom_prompt(self, mock_post):
        """Test continue_request with custom prompt function."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")
        history = ChatHistory()
        history.add_user("Original prompt")

        # Mock initial result
        initial_result = ChatResult(
            text="Partial",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="length",
        )

        # Mock continue response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }
        mock_post.return_value.raise_for_status = lambda: None

        # Custom prompt function
        prompt_calls = []

        def custom_prompt(count, max_count, current_text, original_prompt):
            prompt_calls.append((count, max_count))
            return f"Continue {count}/{max_count}"

        result = ChatContinue.continue_request(
            chat,
            initial_result,
            history=history,
            continue_prompt=custom_prompt,
        )

        # Verify custom prompt was called
        assert len(prompt_calls) > 0
        assert result.finish_reason == "stop"

    def test_continue_request_history_immutability(self, mock_post):
        """Test that continue_request does not modify original history."""
        chat = Chat(base_url="https://api.test.com/v1", api_key="test", model="test")

        # Create original history
        original_history = ChatHistory()
        original_history.add_user("Original message")
        original_messages_count = len(original_history.messages)

        # Mock initial result
        initial_result = ChatResult(
            text="Partial",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="length",
        )

        # Mock continue response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": " complete"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }
        mock_post.return_value.raise_for_status = lambda: None

        result = ChatContinue.continue_request(
            chat,
            initial_result,
            history=original_history,
        )

        # Verify original history is not modified
        assert len(original_history.messages) == original_messages_count
        assert original_history.messages == [{"role": "user", "content": "Original message"}]


class TestNeedsContinue:
    """Test ChatContinue.needs_continue() helper method."""

    def test_needs_continue_true(self):
        """Test needs_continue returns True for length finish_reason."""
        result = ChatResult(
            text="Partial",
            usage=Usage(),
            finish_reason="length",
        )
        assert ChatContinue.needs_continue(result) is True

    def test_needs_continue_false(self):
        """Test needs_continue returns False for non-length finish_reason."""
        result = ChatResult(
            text="Complete",
            usage=Usage(),
            finish_reason="stop",
        )
        assert ChatContinue.needs_continue(result) is False


@pytest.fixture
def mock_post():
    """Mock requests.post for testing."""
    with patch("lexilux.chat.client.requests.post") as mock:
        yield mock
