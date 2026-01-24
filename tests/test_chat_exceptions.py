"""Test chat-specific exceptions"""
import pytest
from lexilux.chat.exceptions import (
    ChatStreamInterruptedError,
    ChatIncompleteResponseError,
)
from lexilux import ChatResult, ChatStreamChunk
from lexilux.usage import Usage


class TestChatStreamInterruptedError:
    """Test ChatStreamInterruptedError"""

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        partial_result = ChatResult(
            text="Partial",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )
        error = ChatStreamInterruptedError(
            message="Stream interrupted",
            partial_result=partial_result,
            received_chunks=[],
            original_error=Exception("Connection lost"),
        )
        assert str(error) == "Stream interrupted"
        assert error.partial_result == partial_result
        assert error.received_chunks == []
        assert error.original_error is not None
        assert str(error.original_error) == "Connection lost"

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters"""
        error = ChatStreamInterruptedError(message="Interrupted")
        assert str(error) == "Interrupted"
        assert error.partial_result is None
        assert error.received_chunks == []
        assert error.original_error is None

    def test_get_partial_text_from_result(self):
        """Test getting partial text from partial_result"""
        partial_result = ChatResult(
            text="Partial text",
            usage=Usage(),
            finish_reason="length",
        )
        error = ChatStreamInterruptedError(
            message="Interrupted",
            partial_result=partial_result,
        )
        assert error.get_partial_text() == "Partial text"

    def test_get_partial_text_from_chunks(self):
        """Test getting partial text from chunks when no result"""
        chunks = [
            ChatStreamChunk(delta="Hello", done=False, finish_reason=None, usage=Usage()),
            ChatStreamChunk(delta=" World", done=False, finish_reason=None, usage=Usage()),
        ]
        error = ChatStreamInterruptedError(
            message="Interrupted",
            received_chunks=chunks,
        )
        assert error.get_partial_text() == "Hello World"

    def test_get_partial_result_from_result(self):
        """Test getting partial result from partial_result"""
        partial_result = ChatResult(
            text="Partial text",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )
        error = ChatStreamInterruptedError(
            message="Interrupted",
            partial_result=partial_result,
        )
        result = error.get_partial_result()
        assert result == partial_result

    def test_get_partial_result_from_chunks(self):
        """Test getting partial result from chunks when no result"""
        chunks = [
            ChatStreamChunk(delta="Hello", done=False, finish_reason=None, usage=Usage()),
        ]
        error = ChatStreamInterruptedError(
            message="Interrupted",
            received_chunks=chunks,
        )
        result = error.get_partial_result()
        assert result.text == "Hello"
        assert result.finish_reason == "length"


class TestChatIncompleteResponseError:
    """Test ChatIncompleteResponseError"""

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        final_result = ChatResult(
            text="Incomplete",
            usage=Usage(input_tokens=10, output_tokens=50, total_tokens=60),
            finish_reason="length",
        )
        error = ChatIncompleteResponseError(
            message="Still incomplete",
            final_result=final_result,
            continue_count=3,
            max_continues=5,
        )
        assert str(error) == "Still incomplete"
        assert error.final_result == final_result
        assert error.continue_count == 3
        assert error.max_continues == 5

    def test_get_final_text(self):
        """Test getting final text"""
        final_result = ChatResult(
            text="Incomplete text",
            usage=Usage(),
            finish_reason="length",
        )
        error = ChatIncompleteResponseError(
            message="Incomplete",
            final_result=final_result,
            continue_count=2,
            max_continues=3,
        )
        assert error.get_final_text() == "Incomplete text"
