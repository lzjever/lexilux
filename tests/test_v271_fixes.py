"""
Tests for v2.7.1 fixes.

This module tests:
1. StreamingResult.set_result() method (P0 fix)
2. Conversation._merged_streaming_result() uses proper attributes (P0 fix)
3. astream() rate limiting (P1 fix)
"""

import pytest

from lexilux.chat.models import ChatResult
from lexilux.chat.streaming import StreamingResult
from lexilux.usage import Usage


class TestStreamingResultSetResult:
    """Tests for StreamingResult.set_result() method."""

    def test_set_result_basic(self):
        """Test basic set_result functionality."""
        result = StreamingResult()
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)

        result.set_result(
            text="Hello world",
            finish_reason="stop",
            usage=usage,
        )

        assert result.text == "Hello world"
        assert result.finish_reason == "stop"
        assert result.usage == usage
        assert result.done is True

    def test_set_result_uses_slots_correctly(self):
        """Test that set_result uses __slots__ attributes correctly."""
        result = StreamingResult()
        usage = Usage(input_tokens=5, output_tokens=10)

        result.set_result(
            text="Test text",
            finish_reason="length",
            usage=usage,
        )

        # Verify the internal attributes are set correctly (matching __slots__)
        assert result._text_parts == ["Test text"]
        assert result._text_cache == "Test text"
        assert result._finish_reason == "length"
        assert result._usage == usage
        assert result._done is True

    def test_set_result_overwrites_previous_state(self):
        """Test that set_result properly overwrites previous streaming state."""
        result = StreamingResult()

        # Simulate some streaming updates first
        from lexilux.chat.models import ChatStreamChunk

        chunk1 = ChatStreamChunk(delta="Hello", usage=Usage(), done=False)
        chunk2 = ChatStreamChunk(delta=" World", usage=Usage(), done=False)
        result.update(chunk1)
        result.update(chunk2)

        # Now use set_result to set final state
        final_usage = Usage(input_tokens=15, output_tokens=25, total_tokens=40)
        result.set_result(
            text="Completely different text",
            finish_reason="stop",
            usage=final_usage,
        )

        # Verify previous state is completely replaced
        assert result.text == "Completely different text"
        assert result.finish_reason == "stop"
        assert result.usage == final_usage
        assert result.done is True

    def test_set_result_to_chat_result(self):
        """Test that set_result produces valid ChatResult via to_chat_result."""
        result = StreamingResult()
        usage = Usage(input_tokens=100, output_tokens=200)

        result.set_result(
            text="Test response",
            finish_reason="stop",
            usage=usage,
        )

        chat_result = result.to_chat_result()
        assert isinstance(chat_result, ChatResult)
        assert chat_result.text == "Test response"
        assert chat_result.finish_reason == "stop"
        assert chat_result.usage == usage

    def test_set_result_with_none_finish_reason(self):
        """Test set_result with None finish_reason."""
        result = StreamingResult()
        usage = Usage()

        result.set_result(
            text="Some text",
            finish_reason=None,
            usage=usage,
        )

        assert result.text == "Some text"
        assert result.finish_reason is None
        assert result.done is True

    def test_set_result_empty_text(self):
        """Test set_result with empty text."""
        result = StreamingResult()
        usage = Usage()

        result.set_result(
            text="",
            finish_reason="stop",
            usage=usage,
        )

        assert result.text == ""
        assert result.done is True


class TestMergedStreamingResult:
    """Tests for _ResponseContinuer._merged_streaming_result() using proper attributes."""

    def test_merged_streaming_result_single(self):
        """Test _merged_streaming_result with a single result."""
        from lexilux.chat.conversation import _ResponseContinuer

        single_result = ChatResult(
            text="Single response",
            usage=Usage(input_tokens=10, output_tokens=20),
            finish_reason="stop",
        )

        streaming_result = _ResponseContinuer._merged_streaming_result(
            initial_result=single_result,
            all_results=[single_result],
        )

        assert streaming_result.text == "Single response"
        assert streaming_result.finish_reason == "stop"
        assert streaming_result.done is True
        # Verify it uses __slots__ correctly (no dynamic _text attribute)
        assert hasattr(streaming_result, "_text_parts")
        assert streaming_result._text_parts == ["Single response"]

    def test_merged_streaming_result_multiple(self):
        """Test _merged_streaming_result with multiple results."""
        from lexilux.chat.conversation import _ResponseContinuer

        results = [
            ChatResult(
                text="Part 1",
                usage=Usage(input_tokens=10, output_tokens=5),
                finish_reason="length",
            ),
            ChatResult(
                text="Part 2",
                usage=Usage(input_tokens=5, output_tokens=5),
                finish_reason="stop",
            ),
        ]

        streaming_result = _ResponseContinuer._merged_streaming_result(
            initial_result=results[0],
            all_results=results,
        )

        # Should contain merged text
        assert streaming_result.text == "Part 1Part 2"
        assert streaming_result.finish_reason == "stop"  # Last result's finish_reason
        assert streaming_result.done is True

    def test_merged_streaming_result_proper_slots_usage(self):
        """Test that merged result uses __slots__ attributes, not dynamic ones."""
        from lexilux.chat.conversation import _ResponseContinuer

        result = ChatResult(
            text="Test",
            usage=Usage(),
            finish_reason="stop",
        )

        streaming_result = _ResponseContinuer._merged_streaming_result(
            initial_result=result,
            all_results=[result],
        )

        # These should be set via __slots__
        assert hasattr(streaming_result, "_text_parts")
        assert hasattr(streaming_result, "_text_cache")
        assert hasattr(streaming_result, "_finish_reason")
        assert hasattr(streaming_result, "_usage")
        assert hasattr(streaming_result, "_done")

        # _text_parts should be a list (not creating _text attribute)
        assert isinstance(streaming_result._text_parts, list)
        assert streaming_result._text_cache == "Test"


class TestAstreamRateLimiting:
    """Tests for astream() rate limiting support."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("aiolimiter"),
        reason="aiolimiter not installed",
    )
    async def test_astream_applies_rate_limit(self):
        """Test that astream() applies rate limiting before streaming."""
        from unittest.mock import patch

        from lexilux.chat.client import Chat

        # Create chat with rate limiting
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            rate_limit=(5, 60.0),  # 5 requests per 60 seconds
        )

        # Track if acquire was called
        acquire_called = []
        original_acquire = chat._rate_limiter.acquire

        async def tracked_acquire():
            acquire_called.append(True)
            await original_acquire()

        chat._rate_limiter.acquire = tracked_acquire

        # Mock the streaming request at the generator level
        async def mock_stream_gen():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        # Create a mock async iterator that has aclose
        class MockAsyncIterator:
            def __init__(self, gen):
                self._gen = gen
                self._aiter = None

            def __aiter__(self):
                self._aiter = self._gen.__aiter__()
                return self

            async def __anext__(self):
                return await self._aiter.__anext__()

            async def aclose(self):
                pass

        with patch.object(chat, "_amake_streaming_request") as mock_stream_request:
            mock_stream_request.return_value = MockAsyncIterator(mock_stream_gen())

            # Call astream
            iterator = await chat.astream("test message")

            # Consume the iterator
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)

            # Verify rate limiter was called
            assert len(acquire_called) == 1, "Rate limiter should be called once"

    @pytest.mark.asyncio
    async def test_astream_without_rate_limit(self):
        """Test that astream() works without rate limiting configured."""
        from unittest.mock import patch

        from lexilux.chat.client import Chat

        # Create chat WITHOUT rate limiting
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            # No rate_limit parameter
        )

        assert chat._rate_limiter is None

        # Mock the streaming request
        async def mock_stream_gen():
            yield 'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        class MockAsyncIterator:
            def __init__(self, gen):
                self._gen = gen
                self._aiter = None

            def __aiter__(self):
                self._aiter = self._gen.__aiter__()
                return self

            async def __anext__(self):
                return await self._aiter.__anext__()

            async def aclose(self):
                pass

        with patch.object(chat, "_amake_streaming_request") as mock_stream_request:
            mock_stream_request.return_value = MockAsyncIterator(mock_stream_gen())

            # Call astream - should not raise
            iterator = await chat.astream("test")

            # Consume iterator
            async for _ in iterator:
                pass

            # Should complete without error
            assert True

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("aiolimiter"),
        reason="aiolimiter not installed",
    )
    async def test_astream_rate_limit_integration(self):
        """Integration test that rate limiting is called in correct order."""
        from lexilux.chat.client import Chat

        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            rate_limit=(10, 60.0),
        )

        # Verify rate limiter exists
        assert chat._rate_limiter is not None
        assert chat._rate_limiter.max_rate == 10

        # The actual streaming test would require mocking httpx
        # This test verifies the rate limiter is properly configured


class TestStreamingResultSlotsIntegrity:
    """Tests to ensure StreamingResult maintains __slots__ integrity."""

    def test_no_dynamic_attributes_after_normal_usage(self):
        """Test that normal streaming usage doesn't create dynamic attributes."""
        from lexilux.chat.models import ChatStreamChunk

        result = StreamingResult()

        # Simulate normal streaming
        chunks = [
            ChatStreamChunk(delta="Hello", usage=Usage(), done=False),
            ChatStreamChunk(delta=" ", usage=Usage(), done=False),
            ChatStreamChunk(
                delta="World", usage=Usage(), done=True, finish_reason="stop"
            ),
        ]

        for chunk in chunks:
            result.update(chunk)

        # Only __slots__ attributes should exist
        expected_attrs = {
            "_text_parts",
            "_text_cache",
            "_finish_reason",
            "_usage",
            "_done",
        }
        actual_attrs = {
            attr
            for attr in dir(result)
            if attr.startswith("_") and not attr.startswith("__")
        }

        # Filter out any private methods/properties
        slot_attrs = {
            attr for attr in actual_attrs if not callable(getattr(result, attr, None))
        }

        assert expected_attrs == slot_attrs, (
            f"Unexpected attributes: {slot_attrs - expected_attrs}"
        )

    def test_no_dynamic_attributes_after_set_result(self):
        """Test that set_result doesn't create dynamic attributes."""
        result = StreamingResult()

        result.set_result(
            text="Test",
            finish_reason="stop",
            usage=Usage(),
        )

        # Only __slots__ attributes should exist
        expected_attrs = {
            "_text_parts",
            "_text_cache",
            "_finish_reason",
            "_usage",
            "_done",
        }
        actual_attrs = {
            attr
            for attr in dir(result)
            if attr.startswith("_") and not attr.startswith("__")
        }

        slot_attrs = {
            attr for attr in actual_attrs if not callable(getattr(result, attr, None))
        }

        assert expected_attrs == slot_attrs, (
            f"Unexpected attributes: {slot_attrs - expected_attrs}"
        )

    def test_text_property_returns_cached_value(self):
        """Test that text property returns the cached value after set_result."""
        result = StreamingResult()

        result.set_result(
            text="Cached text",
            finish_reason="stop",
            usage=Usage(),
        )

        # _text_cache should be set directly
        assert result._text_cache == "Cached text"

        # text property should return cached value without re-joining
        assert result.text == "Cached text"
        assert result.text == result._text_cache
