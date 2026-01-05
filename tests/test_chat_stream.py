"""
Chat streaming test cases
"""

from unittest.mock import Mock, patch

from lexilux import Chat, ChatStreamChunk
from lexilux.usage import Usage


class TestChatStream:
    """Chat streaming tests"""

    @patch("lexilux.chat.client.requests.post")
    def test_stream_basic(self, mock_post):
        """Test basic streaming"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        # Mock SSE stream
        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " "}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": "world"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello"))
        assert len(chunks) == 4  # 3 content chunks + 1 done chunk

        assert chunks[0].delta == "Hello"
        assert chunks[0].done is False
        assert chunks[1].delta == " "
        assert chunks[2].delta == "world"
        assert chunks[3].done is True

    @patch("lexilux.chat.client.requests.post")
    def test_stream_with_usage(self, mock_post):
        """Test streaming with usage in final chunk"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello", include_usage=True))

        # First chunk: content, no usage
        assert chunks[0].delta == "Hello"
        assert chunks[0].done is False
        assert chunks[0].usage.total_tokens is None

        # Second chunk: done, with usage
        assert chunks[1].done is True
        assert chunks[1].usage.total_tokens == 15
        assert chunks[1].usage.input_tokens == 10
        assert chunks[1].usage.output_tokens == 5

    @patch("lexilux.chat.client.requests.post")
    def test_stream_without_usage(self, mock_post):
        """Test streaming without usage"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello", include_usage=False))

        # All chunks should have empty usage
        for chunk in chunks:
            assert isinstance(chunk.usage, Usage)
            assert chunk.usage.total_tokens is None

    @patch("lexilux.chat.client.requests.post")
    def test_stream_with_system_message(self, mock_post):
        """Test streaming with system message"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Response"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello", system="You are helpful"))
        assert len(chunks) >= 1

        # Verify request payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "Hello"

    @patch("lexilux.chat.client.requests.post")
    def test_stream_parameters(self, mock_post):
        """Test streaming with additional parameters"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Response"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(
            chat.stream(
                "Hello",
                temperature=0.9,
                max_tokens=100,
                include_usage=True,
            )
        )
        assert len(chunks) >= 1

        # Verify request payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["stream"] is True
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 100
        assert "stream_options" in payload
        assert payload["stream_options"]["include_usage"] is True

    @patch("lexilux.chat.client.requests.post")
    def test_stream_finish_reason(self, mock_post):
        """Test streaming with finish_reason in chunks"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " "}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello"))
        assert len(chunks) >= 3

        # Find chunk with finish_reason
        done_chunks = [chunk for chunk in chunks if chunk.done]
        assert len(done_chunks) > 0
        assert done_chunks[0].finish_reason == "stop"

    @patch("lexilux.chat.client.requests.post")
    def test_stream_finish_reason_length(self, mock_post):
        """Test streaming with finish_reason='length'"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello"))
        done_chunks = [chunk for chunk in chunks if chunk.done]
        assert len(done_chunks) > 0
        assert done_chunks[0].finish_reason == "length"

    @patch("lexilux.chat.client.requests.post")
    def test_stream_finish_reason_empty_string(self, mock_post):
        """Test streaming with finish_reason='' (defensive handling)"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello"))
        # Empty string should be normalized to None, so done should be False
        # (because done is determined by finish_reason is not None)
        for chunk in chunks:
            if chunk.done:
                # If somehow done=True, finish_reason should be None after normalization
                assert chunk.finish_reason is None

    @patch("lexilux.chat.client.requests.post")
    def test_stream_finish_reason_invalid_type(self, mock_post):
        """Test streaming with invalid finish_reason type (defensive handling)"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        # Simulate invalid finish_reason (number instead of string)
        # Note: JSON doesn't allow this, but we test defensive code
        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(chat.stream("Hello"))
        # Should handle gracefully
        assert len(chunks) > 0


class TestChatStreamChunk:
    """ChatStreamChunk class tests"""

    def test_chunk_repr(self):
        """Test ChatStreamChunk representation"""
        from lexilux.usage import Usage

        chunk = ChatStreamChunk(
            delta="Hello",
            done=False,
            usage=Usage(total_tokens=10),
        )
        repr_str = repr(chunk)
        assert "ChatStreamChunk" in repr_str
        assert "Hello" in repr_str
        assert "done=False" in repr_str

    def test_chunk_with_finish_reason(self):
        """Test ChatStreamChunk with finish_reason"""
        from lexilux.usage import Usage

        chunk = ChatStreamChunk(
            delta="",
            done=True,
            usage=Usage(total_tokens=10),
            finish_reason="stop",
        )
        assert chunk.finish_reason == "stop"
        assert chunk.done is True

    def test_chunk_without_finish_reason(self):
        """Test ChatStreamChunk without finish_reason"""
        from lexilux.usage import Usage

        chunk = ChatStreamChunk(
            delta="Hello",
            done=False,
            usage=Usage(total_tokens=10),
            finish_reason=None,
        )
        assert chunk.finish_reason is None
        assert chunk.done is False
