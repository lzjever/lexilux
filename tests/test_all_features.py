#!/usr/bin/env python
"""Comprehensive test for all refactor plan features"""

import pytest
import responses

from lexilux import Chat, ChatContinue, ChatHistory, ChatHistoryFormatter, StreamingIterator
from lexilux.chat.history import filter_by_role, merge_histories, search_content, get_statistics


@responses.activate
def test_chat_with_history():
    """Test chat_with_history method"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
    )

    history = ChatHistory.from_messages("Hello")

    # Mock API response
    responses.add(
        responses.POST,
        "https://api.example.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        status=200,
    )

    result = chat.chat_with_history(history, temperature=0.7)
    assert result.text == "Hi!"
    assert result.usage.total_tokens == 15


@responses.activate
def test_stream_with_history():
    """Test stream_with_history method"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
    )

    history = ChatHistory.from_messages("Hello")

    # Mock streaming response
    stream_data = [
        b'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}\n',
        b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
        b"data: [DONE]\n",
    ]

    responses.add(
        responses.POST,
        "https://api.example.com/v1/chat/completions",
        body=b''.join(stream_data),
        status=200,
        content_type="text/event-stream",
    )

    result = chat.stream_with_history(history)
    assert isinstance(result, StreamingIterator)
    chunks = list(result)
    assert len(chunks) >= 1


@responses.activate
def test_chat_with_additional_message():
    """Test chat_with_history with additional message"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
    )

    history = ChatHistory.from_messages("First message")
    history.add_assistant("First response")

    # Mock API response
    responses.add(
        responses.POST,
        "https://api.example.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "Second response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        },
        status=200,
    )

    result = chat.chat_with_history(history, "Second message", temperature=0.7)
    assert result.text == "Second response"
    assert result.usage.total_tokens == 30


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


def test_history_formatter():
    """Test ChatHistoryFormatter"""
    history = ChatHistory.from_messages("Hello", system="You are a helpful assistant")
    history.add_assistant("Hi! How can I help you?")

    # Test to_text
    text_output = ChatHistoryFormatter.to_text(history)
    assert "Hello" in text_output
    assert "Hi!" in text_output

    # Test to_markdown
    markdown_output = ChatHistoryFormatter.to_markdown(history)
    assert "Hello" in markdown_output

    # Test to_json
    json_output = ChatHistoryFormatter.to_json(history)
    assert isinstance(json_output, str)

    # Test to_html
    html_output = ChatHistoryFormatter.to_html(history)
    assert "Hello" in html_output


@responses.activate
def test_history_tracking():
    """Test history tracking in chat calls"""
    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4",
    )

    # First call
    responses.add(
        responses.POST,
        "https://api.example.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "Hello! How can I help?"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
        status=200,
    )

    result1 = chat("Hello")

    # Verify we can create history from result
    history = ChatHistory.from_chat_result("Hello", result1)
    assert len(history.messages) == 2  # user + assistant
    assert history.messages[0]["role"] == "user"
    assert history.messages[1]["role"] == "assistant"
    
    # Can also create with system separately - system is not in messages list
    history_with_system = ChatHistory(system="System message")
    history_with_system.add_user("Hello")
    history_with_system.add_assistant(result1.text)
    assert len(history_with_system.messages) == 2  # user + assistant (system is separate attribute)
    assert history_with_system.system == "System message"
    assert history_with_system.messages[0]["role"] == "user"
    
    # get_messages includes system message
    all_messages = history_with_system.get_messages()
    assert len(all_messages) == 3  # system + user + assistant
    assert all_messages[0]["role"] == "system"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
