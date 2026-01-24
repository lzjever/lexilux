"""Test reasoning_content field support for OpenAI o1/Claude 3.5/DeepSeek R1."""

import pytest
from lexilux.chat.models import ChatStreamChunk
from lexilux.chat._request import SSEChatStreamParser
from lexilux.usage import Usage


def test_reasoning_content_in_chunk():
    """Test ChatStreamChunk with reasoning_content."""
    chunk = ChatStreamChunk(
        delta="Hello",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        done=False,
        reasoning_content="I need to think about this...",
        reasoning_tokens=25,
    )

    assert chunk.reasoning_content == "I need to think about this..."
    assert chunk.reasoning_tokens == 25
    assert "reasoning=True" in repr(chunk)


def test_chunk_without_reasoning():
    """Test ChatStreamChunk without reasoning_content."""
    chunk = ChatStreamChunk(
        delta="Hello",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        done=False,
    )

    assert chunk.reasoning_content is None
    assert chunk.reasoning_tokens is None
    assert "reasoning=True" not in repr(chunk)


def test_sse_parser_with_reasoning_disabled():
    """Test SSE parser without reasoning support."""
    parser = SSEChatStreamParser(
        return_raw_events=False,
        include_reasoning=False,  # Disabled
    )

    # Simulate chunk with reasoning content
    line = 'data: {"choices": [{"delta": {"content": "Hello", "reasoning_content": "Thinking..."}}]}'
    chunk = parser.feed_line(line)

    assert chunk is not None
    assert chunk.delta == "Hello"
    assert chunk.reasoning_content is None  # Should be ignored when disabled


def test_sse_parser_with_reasoning_enabled():
    """Test SSE parser handles reasoning content when enabled."""
    parser = SSEChatStreamParser(return_raw_events=False, include_reasoning=True)

    # Simulate reasoning chunk
    line = 'data: {"choices": [{"delta": {"content": "Hello", "reasoning_content": "Thinking...", "reasoning_tokens": 15}}]}'
    chunk = parser.feed_line(line)

    assert chunk is not None
    assert chunk.delta == "Hello"
    assert chunk.reasoning_content == "Thinking..."
    assert chunk.reasoning_tokens == 15


def test_sse_parser_done_chunk():
    """Test SSE parser [DONE] chunk maintains consistency."""
    parser = SSEChatStreamParser(return_raw_events=False, include_reasoning=True)

    # Simulate [DONE] chunk
    line = "data: [DONE]"
    chunk = parser.feed_line(line)

    assert chunk is not None
    assert chunk.done is True
    assert chunk.delta == ""
    assert chunk.reasoning_content is None  # No reasoning in [DONE]
    assert chunk.reasoning_tokens is None


def test_sse_parser_reasoning_only():
    """Test SSE parser with only reasoning content."""
    parser = SSEChatStreamParser(return_raw_events=False, include_reasoning=True)

    # Simulate chunk with only reasoning, no regular content
    line = 'data: {"choices": [{"delta": {"reasoning_content": "Let me think..."}}]}'
    chunk = parser.feed_line(line)

    assert chunk is not None
    assert chunk.delta == ""  # No regular content
    assert chunk.reasoning_content == "Let me think..."


if __name__ == "__main__":
    pytest.main([__file__])
