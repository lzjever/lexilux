"""
Tests for async API support.

This module tests the async versions of Chat, Embed, and Rerank methods.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lexilux import (
    AsyncStreamingIterator,
    Chat,
    ChatResult,
    ChatStreamChunk,
    Embed,
    EmbedResult,
    Rerank,
    RerankResult,
)
from lexilux.chat.history import ChatHistory
from lexilux.exceptions import ValidationError
from lexilux.usage import Usage


# ============================================================================
# Chat Async Tests
# ============================================================================


class TestChatAcall:
    """Tests for Chat.acall() async method."""

    @pytest.fixture
    def chat(self):
        """Create a Chat instance for testing."""
        return Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

    @pytest.fixture
    def mock_response_data(self):
        """Mock API response data."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    @pytest.mark.asyncio
    async def test_acall_basic(self, chat, mock_response_data):
        """Test basic async call."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch.object(chat, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await chat.acall("Hello")

            assert isinstance(result, ChatResult)
            assert result.text == "Hello! How can I help?"
            assert result.finish_reason == "stop"
            assert result.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_acall_with_history(self, chat, mock_response_data):
        """Test async call with history."""
        history = ChatHistory()
        history.add_user("Previous message")
        history.add_assistant("Previous response")

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch.object(chat, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await chat.acall("New message", history=history)

            assert isinstance(result, ChatResult)
            # Original history should not be modified
            assert len(history.messages) == 2

    @pytest.mark.asyncio
    async def test_acall_requires_model(self, chat):
        """Test that acall requires a model."""
        chat.model = None
        with pytest.raises(ValidationError, match="Model must be specified"):
            await chat.acall("Hello")


class TestChatAstream:
    """Tests for Chat.astream() async method."""

    @pytest.fixture
    def chat(self):
        """Create a Chat instance for testing."""
        return Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

    @pytest.mark.asyncio
    async def test_astream_returns_async_iterator(self, chat):
        """Test that astream returns an AsyncStreamingIterator."""

        # Mock the streaming request
        async def mock_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"!"}, "finish_reason":"stop"}]}'
            yield "data: [DONE]"

        with patch.object(chat, "_amake_streaming_request", mock_stream):
            iterator = await chat.astream("Hello")
            assert isinstance(iterator, AsyncStreamingIterator)

    @pytest.mark.asyncio
    async def test_astream_accumulates_text(self, chat):
        """Test that astream accumulates text correctly."""

        async def mock_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"}, "finish_reason":"stop"}]}'
            yield "data: [DONE]"

        with patch.object(chat, "_amake_streaming_request", mock_stream):
            iterator = await chat.astream("Hello")
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)

            assert len(chunks) >= 2
            assert iterator.result.text == "Hello World"

    @pytest.mark.asyncio
    async def test_astream_with_history(self, chat):
        """Test astream with history - history is immutable (cloned)."""
        history = ChatHistory()
        history.add_user("Previous message")
        original_message_count = len(history.messages)

        async def mock_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"!"}, "finish_reason":"stop"}]}'

        with patch.object(chat, "_amake_streaming_request", mock_stream):
            iterator = await chat.astream("Hello", history=history)
            async for _ in iterator:
                pass

            # Original history is NOT modified (immutable behavior)
            assert len(history.messages) == original_message_count
            assert history.messages[0]["role"] == "user"
            assert history.messages[0]["content"] == "Previous message"

            # Verify that streaming result still includes history context
            assert iterator.result.text == "Hello!"


class TestChatAcomplete:
    """Tests for Chat.acomplete() async method."""

    @pytest.fixture
    def chat(self):
        """Create a Chat instance for testing."""
        return Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

    @pytest.mark.asyncio
    async def test_acomplete_non_truncated(self, chat):
        """Test acomplete with non-truncated response."""
        mock_response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Complete response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch.object(chat, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await chat.acomplete("Hello")

            assert isinstance(result, ChatResult)
            assert result.text == "Complete response"
            assert result.finish_reason == "stop"


# ============================================================================
# Embed Async Tests
# ============================================================================


class TestEmbedAcall:
    """Tests for Embed.acall() async method."""

    @pytest.fixture
    def embed(self):
        """Create an Embed instance for testing."""
        return Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

    @pytest.fixture
    def mock_response_data(self):
        """Mock API response data."""
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    @pytest.mark.asyncio
    async def test_acall_single_text(self, embed, mock_response_data):
        """Test async embedding of single text."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        with patch.object(embed, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await embed.acall("Hello")

            assert isinstance(result, EmbedResult)
            assert result.vectors == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_acall_multiple_texts(self, embed):
        """Test async embedding of multiple texts."""
        mock_response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        with patch.object(embed, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await embed.acall(["Hello", "World"])

            assert isinstance(result, EmbedResult)
            assert len(result.vectors) == 2

    @pytest.mark.asyncio
    async def test_acall_empty_input_error(self, embed):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            await embed.acall([])


# ============================================================================
# Rerank Async Tests
# ============================================================================


class TestRerankAcall:
    """Tests for Rerank.acall() async method."""

    @pytest.fixture
    def rerank(self):
        """Create a Rerank instance for testing."""
        return Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
        )

    @pytest.fixture
    def mock_response_data(self):
        """Mock API response data."""
        return {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 2, "relevance_score": 0.75},
            ],
            "usage": {"total_tokens": 50},
        }

    @pytest.mark.asyncio
    async def test_acall_basic(self, rerank, mock_response_data):
        """Test basic async rerank."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        with patch.object(rerank, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await rerank.acall("python http", ["urllib", "requests", "httpx"])

            assert isinstance(result, RerankResult)
            assert len(result.results) == 3
            # Results should be sorted by score descending
            assert result.results[0][1] >= result.results[1][1]

    @pytest.mark.asyncio
    async def test_acall_empty_docs_error(self, rerank):
        """Test that empty docs raises ValueError."""
        with pytest.raises(ValueError, match="Docs cannot be empty"):
            await rerank.acall("query", [])


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestAsyncContextManagers:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_chat_async_context_manager(self):
        """Test Chat async context manager."""
        async with Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        ) as chat:
            assert chat is not None

    @pytest.mark.asyncio
    async def test_embed_async_context_manager(self):
        """Test Embed async context manager."""
        async with Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        ) as embed:
            assert embed is not None

    @pytest.mark.asyncio
    async def test_rerank_async_context_manager(self):
        """Test Rerank async context manager."""
        async with Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
        ) as rerank:
            assert rerank is not None


# ============================================================================
# AsyncStreamingIterator Tests
# ============================================================================


class TestAsyncStreamingIterator:
    """Tests for AsyncStreamingIterator class."""

    @pytest.mark.asyncio
    async def test_async_iterator_protocol(self):
        """Test that AsyncStreamingIterator implements async iterator protocol."""

        async def mock_generator():
            yield ChatStreamChunk(delta="Hello", done=False, usage=Usage())
            yield ChatStreamChunk(
                delta=" World", done=True, finish_reason="stop", usage=Usage()
            )

        iterator = AsyncStreamingIterator(mock_generator())
        chunks = []
        async for chunk in iterator:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert iterator.result.text == "Hello World"
        assert iterator.result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_collect_method(self):
        """Test AsyncStreamingIterator.collect() method."""

        async def mock_generator():
            yield ChatStreamChunk(delta="Hello", done=False, usage=Usage())
            yield ChatStreamChunk(
                delta=" World",
                done=True,
                finish_reason="stop",
                usage=Usage(total_tokens=10),
            )

        iterator = AsyncStreamingIterator(mock_generator())
        result = await iterator.collect()

        assert isinstance(result, ChatResult)
        assert result.text == "Hello World"
        assert result.finish_reason == "stop"
