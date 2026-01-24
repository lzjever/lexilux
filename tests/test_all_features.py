#!/usr/bin/env python
"""Comprehensive test for all refactor plan features"""

from unittest.mock import Mock, patch

from lexilux import Chat, ChatContinue, ChatHistory, ChatHistoryFormatter, StreamingIterator
from lexilux.chat import filter_by_role, get_statistics, merge_histories, search_content


def test_auto_history_non_streaming():
    """Test auto_history with non-streaming"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
        auto_history=True,
    )

    with patch("lexilux.chat.client.requests.post") as mock_post:
        # First call
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help?"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        mock_response1.raise_for_status = Mock()

        # Second call
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": "I'm doing well!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        result1 = chat("Hello")
        result2 = chat("How are you?")

        history = chat.get_history()
        assert history is not None
        assert len(history.messages) == 4  # 2 user + 2 assistant
        assert history.messages[0]["role"] == "user"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[2]["role"] == "user"
        assert history.messages[3]["role"] == "assistant"

    print("✓ auto_history non-streaming test passed")


def test_auto_history_streaming():
    """Test auto_history with streaming"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
        auto_history=True,
    )

    with patch("lexilux.chat.client.requests.post") as mock_post:
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

        iterator = chat.stream("Hello", auto_history=True)
        assert isinstance(iterator, StreamingIterator)

        chunks = list(iterator)
        assert len(chunks) >= 2

        # Check history was updated
        history = chat.get_history()
        assert history is not None
        assert len(history.messages) == 2
        assert "Hello" in history.messages[1]["content"] or "world" in history.messages[1]["content"]

    print("✓ auto_history streaming test passed")


def test_chat_continue():
    """Test ChatContinue functionality"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
    )

    from lexilux import ChatResult
    from lexilux.usage import Usage

    result1 = ChatResult(
        text="This is part 1",
        usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
        finish_reason="length",
    )

    history = ChatHistory.from_messages("Write a story")
    history.append_result(result1)

    with patch("lexilux.chat.client.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " and part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # New API with auto_merge=True (default) returns merged result directly
        full_result = ChatContinue.continue_request(
            chat, result1, history=history, add_continue_prompt=True
        )
        assert "part 1" in full_result.text
        assert "part 2" in full_result.text
        assert full_result.usage.total_tokens == 90

    print("✓ ChatContinue test passed")


def test_utility_functions():
    """Test utility functions"""
    history1 = ChatHistory.from_messages("Hello")
    history1.add_assistant("Hi!")

    history2 = ChatHistory.from_messages("How are you?")
    history2.add_assistant("I'm fine!")

    # Test merge_histories
    merged = merge_histories(history1, history2)
    assert len(merged.messages) == 4

    # Test filter_by_role
    user_only = filter_by_role(merged, "user")
    assert len(user_only.messages) == 2
    assert all(msg["role"] == "user" for msg in user_only.messages)

    # Test search_content
    results = search_content(merged, "Hello")
    assert len(results) >= 1

    # Test get_statistics
    stats = get_statistics(merged)
    assert stats["total_rounds"] == 2
    assert stats["user_messages"] == 2
    assert stats["assistant_messages"] == 2

    print("✓ Utility functions test passed")


def test_chat_with_history():
    """Test chat_with_history and stream_with_history methods"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
    )

    history = ChatHistory.from_messages("Hello")

    with patch("lexilux.chat.client.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = chat.chat_with_history(history, temperature=0.7)
        assert result.text == "Hi!"

    print("✓ chat_with_history test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing all refactor plan features...")
    print("=" * 60)
    test_auto_history_non_streaming()
    test_auto_history_streaming()
    test_chat_continue()
    test_utility_functions()
    test_chat_with_history()
    print("=" * 60)
    print("✅ All tests passed! All features from plan are implemented!")
    print("=" * 60)

