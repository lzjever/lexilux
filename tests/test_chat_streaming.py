"""
Comprehensive tests for StreamingResult and StreamingIterator classes.

Tests are written based on the public interface, not implementation details.
"""

from lexilux import Usage
from lexilux.chat import ChatResult, ChatStreamChunk, StreamingIterator, StreamingResult


class TestStreamingResult:
    """Test StreamingResult class"""

    def test_init(self):
        """Test StreamingResult initialization"""
        result = StreamingResult()
        assert result.text == ""
        assert result.finish_reason is None
        assert isinstance(result.usage, Usage)
        assert result.done is False

    def test_update_with_delta(self):
        """Test update with text delta"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
        result.update(chunk)
        assert result.text == "Hello "
        assert result.done is False
        assert result.finish_reason is None

    def test_update_accumulates_text(self):
        """Test that update accumulates text"""
        result = StreamingResult()
        chunk1 = ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
        chunk2 = ChatStreamChunk(delta="world", done=False, usage=Usage(), finish_reason=None)
        result.update(chunk1)
        result.update(chunk2)
        assert result.text == "Hello world"

    def test_update_sets_done(self):
        """Test that update sets done flag"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="Done", done=True, usage=Usage(), finish_reason="stop")
        result.update(chunk)
        assert result.done is True
        assert result.finish_reason == "stop"

    def test_update_sets_finish_reason(self):
        """Test that update sets finish_reason"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="", done=True, usage=Usage(), finish_reason="length")
        result.update(chunk)
        assert result.finish_reason == "length"

    def test_update_sets_usage(self):
        """Test that update sets usage"""
        result = StreamingResult()
        usage = Usage(total_tokens=100, input_tokens=50, output_tokens=50)
        chunk = ChatStreamChunk(delta="", done=True, usage=usage, finish_reason="stop")
        result.update(chunk)
        assert result.usage.total_tokens == 100

    def test_update_empty_delta(self):
        """Test update with empty delta"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="", done=False, usage=Usage(), finish_reason=None)
        result.update(chunk)
        assert result.text == ""
        assert result.done is False

    def test_to_chat_result(self):
        """Test to_chat_result conversion"""
        result = StreamingResult()
        chunk1 = ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
        chunk2 = ChatStreamChunk(
            delta="world",
            done=True,
            usage=Usage(total_tokens=10),
            finish_reason="stop",
        )
        result.update(chunk1)
        result.update(chunk2)
        chat_result = result.to_chat_result()
        assert isinstance(chat_result, ChatResult)
        assert chat_result.text == "Hello world"
        assert chat_result.finish_reason == "stop"
        assert chat_result.usage.total_tokens == 10

    def test_to_chat_result_incomplete(self):
        """Test to_chat_result with incomplete stream"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="Partial", done=False, usage=Usage(), finish_reason=None)
        result.update(chunk)
        chat_result = result.to_chat_result()
        assert chat_result.text == "Partial"
        assert chat_result.finish_reason is None

    def test_str_representation(self):
        """Test string representation"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="Hello", done=True, usage=Usage(), finish_reason="stop")
        result.update(chunk)
        assert str(result) == "Hello"
        assert result.text == "Hello"

    def test_repr_representation(self):
        """Test repr representation"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="Hello", done=True, usage=Usage(), finish_reason="stop")
        result.update(chunk)
        repr_str = repr(result)
        assert "StreamingResult" in repr_str
        assert "Hello" in repr_str
        assert "done=True" in repr_str or "done=True" in repr_str


class TestStreamingIterator:
    """Test StreamingIterator class"""

    def test_init(self):
        """Test StreamingIterator initialization"""

        def empty_chunks():
            return
            yield  # Make it a generator

        iterator = StreamingIterator(empty_chunks())
        assert iterator.result is not None
        assert isinstance(iterator.result, StreamingResult)
        assert iterator.result.text == ""

    def test_iteration(self):
        """Test iterator iteration"""

        def mock_chunks():
            yield ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
            yield ChatStreamChunk(delta="world", done=True, usage=Usage(), finish_reason="stop")

        iterator = StreamingIterator(mock_chunks())
        chunks = list(iterator)
        assert len(chunks) == 2
        assert chunks[0].delta == "Hello "
        assert chunks[1].delta == "world"

    def test_result_updates_during_iteration(self):
        """Test that result updates during iteration"""

        def mock_chunks():
            yield ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
            yield ChatStreamChunk(delta="world", done=True, usage=Usage(), finish_reason="stop")

        iterator = StreamingIterator(mock_chunks())
        # Before iteration
        assert iterator.result.text == ""
        # During iteration
        chunks = []
        for chunk in iterator:
            chunks.append(chunk)
            # Result should be updated
            assert len(iterator.result.text) > 0
        # After iteration
        assert iterator.result.text == "Hello world"
        assert iterator.result.done is True

    def test_result_accessible_during_iteration(self):
        """Test that result is accessible during iteration"""

        def mock_chunks():
            yield ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
            yield ChatStreamChunk(delta="world", done=True, usage=Usage(), finish_reason="stop")

        iterator = StreamingIterator(mock_chunks())
        accumulated = []
        for chunk in iterator:
            # Can access result at any time
            accumulated.append(iterator.result.text)
        assert accumulated[0] == "Hello "
        assert accumulated[1] == "Hello world"

    def test_result_after_empty_iteration(self):
        """Test result after empty iteration"""

        def empty_chunks():
            return
            yield

        iterator = StreamingIterator(empty_chunks())
        list(iterator)  # Consume iterator
        assert iterator.result.text == ""
        assert iterator.result.done is False

    def test_result_with_multiple_chunks(self):
        """Test result with many chunks"""

        def mock_chunks():
            for i in range(10):
                yield ChatStreamChunk(
                    delta=f"chunk{i} ",
                    done=False,
                    usage=Usage(),
                    finish_reason=None,
                )
            yield ChatStreamChunk(delta="final", done=True, usage=Usage(), finish_reason="stop")

        iterator = StreamingIterator(mock_chunks())
        chunks = list(iterator)
        assert len(chunks) == 11
        expected_text = " ".join(f"chunk{i}" for i in range(10)) + " final"
        assert iterator.result.text == expected_text

    def test_result_finish_reason(self):
        """Test result finish_reason after iteration"""

        def mock_chunks():
            yield ChatStreamChunk(delta="Hello", done=True, usage=Usage(), finish_reason="length")

        iterator = StreamingIterator(mock_chunks())
        list(iterator)
        assert iterator.result.finish_reason == "length"
        assert iterator.result.done is True

    def test_result_usage(self):
        """Test result usage after iteration"""
        usage = Usage(total_tokens=100, input_tokens=50, output_tokens=50)

        def mock_chunks():
            yield ChatStreamChunk(delta="Hello", done=True, usage=usage, finish_reason="stop")

        iterator = StreamingIterator(mock_chunks())
        list(iterator)
        assert iterator.result.usage.total_tokens == 100


class TestStreamingEdgeCases:
    """Test streaming edge cases"""

    def test_streaming_result_multiple_updates_same_chunk(self):
        """Test updating result multiple times with same chunk"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="Hello", done=False, usage=Usage(), finish_reason=None)
        result.update(chunk)
        result.update(chunk)  # Update again
        assert result.text == "HelloHello"  # Should accumulate

    def test_streaming_result_done_false_then_true(self):
        """Test result when done changes from False to True"""
        result = StreamingResult()
        chunk1 = ChatStreamChunk(delta="Hello", done=False, usage=Usage(), finish_reason=None)
        chunk2 = ChatStreamChunk(delta=" world", done=True, usage=Usage(), finish_reason="stop")
        result.update(chunk1)
        assert result.done is False
        result.update(chunk2)
        assert result.done is True

    def test_streaming_iterator_partial_consumption(self):
        """Test iterator with partial consumption"""

        def mock_chunks():
            yield ChatStreamChunk(delta="Hello ", done=False, usage=Usage(), finish_reason=None)
            yield ChatStreamChunk(delta="world", done=True, usage=Usage(), finish_reason="stop")

        iterator = StreamingIterator(mock_chunks())
        # Consume only first chunk - need to call __iter__ first
        iter_obj = iter(iterator)
        next(iter_obj)  # Consume first chunk
        assert iterator.result.text == "Hello "
        # Don't consume second chunk
        # Result should still reflect partial state
        assert iterator.result.done is False

    def test_streaming_result_empty_delta_done(self):
        """Test result with empty delta but done=True"""
        result = StreamingResult()
        chunk = ChatStreamChunk(delta="", done=True, usage=Usage(), finish_reason="stop")
        result.update(chunk)
        assert result.text == ""
        assert result.done is True
        assert result.finish_reason == "stop"

    def test_streaming_result_usage_accumulation(self):
        """Test that usage is updated from final chunk"""
        result = StreamingResult()
        chunk1 = ChatStreamChunk(delta="Hello", done=False, usage=Usage(), finish_reason=None)
        chunk2 = ChatStreamChunk(
            delta=" world",
            done=True,
            usage=Usage(total_tokens=100),
            finish_reason="stop",
        )
        result.update(chunk1)
        assert result.usage.total_tokens is None
        result.update(chunk2)
        assert result.usage.total_tokens == 100
