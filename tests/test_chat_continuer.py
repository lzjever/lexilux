"""
Tests for ConversationContinuer functionality.

Tests verify that the continuer correctly handles conversation continuation
with automatic truncation handling.
"""

from unittest.mock import Mock, patch, AsyncMock

import pytest

from lexilux import Chat
from lexilux.chat.continuer import ConversationContinuer
from lexilux.chat.exceptions import ChatIncompleteResponseError
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult
from lexilux.exceptions import ServerError


class TestConversationContinuerInit:
    """Test ConversationContinuer initialization."""

    def test_init_with_chat_client(self):
        """Test initialization with a Chat client."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)
        assert continuer._client is chat


class TestConversationContinuerComplete:
    """Test ConversationContinuer.complete method."""

    @patch("requests.Session.post")
    def test_complete_no_continuation_needed(self, mock_post):
        """Test complete when response is not truncated."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = continuer.complete("Hello")

        assert result.text == "Hello!"
        assert result.finish_reason == "stop"

    @patch("requests.Session.post")
    def test_complete_with_truncated_response(self, mock_post):
        """Test complete handles truncated response by continuing."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            response = Mock()
            response.status_code = 200
            response.raise_for_status = Mock()
            if call_count[0] == 1:
                response.json.return_value = {
                    "choices": [
                        {"message": {"content": "Part 1"}, "finish_reason": "length"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            else:
                response.json.return_value = {
                    "choices": [
                        {"message": {"content": " Part 2"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            return response

        mock_post.side_effect = mock_response

        result = continuer.complete("Write a story", max_continues=1)

        assert "Part 1" in result.text
        assert result.finish_reason == "stop"

    @patch("requests.Session.post")
    def test_complete_max_continues_reached(self, mock_post):
        """Test complete raises error when max_continues exceeded with ensure_complete=True."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Truncated"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        with pytest.raises(ChatIncompleteResponseError):
            continuer.complete("Write a story", max_continues=0, ensure_complete=True)

    @patch("requests.Session.post")
    def test_complete_with_ensure_complete_false(self, mock_post):
        """Test complete returns partial result when ensure_complete=False."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Truncated"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = continuer.complete(
            "Write a story", max_continues=0, ensure_complete=False
        )

        assert result.text == "Truncated"
        assert result.finish_reason == "length"

    @patch("requests.Session.post")
    def test_complete_with_history(self, mock_post):
        """Test complete with ChatHistory."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)
        history = ChatHistory()
        history.add_user("Previous message")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = continuer.complete("New message", history=history)

        assert result.text == "Response"

    @patch("requests.Session.post")
    def test_complete_error_with_return_partial(self, mock_post):
        """Test complete returns partial result on error when on_error='return_partial'."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            response = Mock()
            response.status_code = 200
            response.raise_for_status = Mock()
            if call_count[0] == 1:
                response.json.return_value = {
                    "choices": [
                        {"message": {"content": "Partial"}, "finish_reason": "length"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
                return response
            else:
                raise ServerError("Network error")

        mock_post.side_effect = mock_response

        result = continuer.complete(
            "Write story",
            max_continues=1,
            ensure_complete=False,
            on_error="return_partial",
        )

        assert "Partial" in result.text


class TestConversationContinuerCompleteStream:
    """Test ConversationContinuer.complete_stream method."""

    @patch("requests.Session.post")
    def test_complete_stream_yields_chunks(self, mock_post):
        """Test complete_stream yields stream chunks."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        # Mock the streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            b'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}',
            b"data: [DONE]",
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        iterator = continuer.complete_stream("Hello")

        chunks = list(iterator)
        assert len(chunks) >= 1

    @patch("requests.Session.post")
    def test_complete_stream_final_chunk_done(self, mock_post):
        """Test complete_stream sets done flag on final chunk."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}',
            b"data: [DONE]",
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        iterator = continuer.complete_stream("Hello")

        # Consume the iterator
        list(iterator)

        # Check that the result is marked as done
        assert iterator.result._done is True


class TestConversationContinuerAsync:
    """Test ConversationContinuer async methods."""

    @pytest.mark.asyncio
    async def test_acomplete_basic(self):
        """Test acomplete basic functionality."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch.object(chat, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await continuer.acomplete("Hello")

        assert result.text == "Hello!"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_acomplete_with_truncation(self):
        """Test acomplete handles truncation."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        call_count = [0]

        def mock_response_factory(*args, **kwargs):
            call_count[0] += 1
            response = Mock()
            response.is_success = True
            response.status_code = 200
            if call_count[0] == 1:
                response.json.return_value = {
                    "choices": [
                        {"message": {"content": "Part 1"}, "finish_reason": "length"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            else:
                response.json.return_value = {
                    "choices": [
                        {"message": {"content": " Part 2"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            return response

        with patch.object(chat, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = mock_response_factory
            mock_get_client.return_value = mock_client

            result = await continuer.acomplete("Write a story", max_continues=1)

        assert "Part 1" in result.text

    @pytest.mark.asyncio
    async def test_acomplete_returns_result(self):
        """Test acomplete returns ChatResult."""
        chat = Chat(
            base_url="https://api.example.com/v1", api_key="test", model="gpt-4"
        )
        continuer = ConversationContinuer(chat)

        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch.object(chat, "_get_async_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await continuer.acomplete("Test message")

        assert isinstance(result, ChatResult)
        assert result.text == "Response"
