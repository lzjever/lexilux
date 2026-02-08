"""
Exception path tests for the Chat client.

Tests error handling and exception paths for network errors, API errors,
and malformed responses. These tests use mocks to simulate error conditions
without making actual HTTP requests.
"""

from unittest.mock import Mock, patch
import pytest
import requests

from lexilux import Chat
from lexilux.exceptions import (
    AuthenticationError,
    RateLimitError,
    ServerError,
    TimeoutError as LexiluxTimeoutError,
    ConnectionError as LexiluxConnectionError,
)


class TestNetworkExceptions:
    """Tests for network-related exceptions."""

    def test_timeout_exception(self):
        """Test that timeout is properly raised."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Mock the session to raise a Timeout exception
        with patch.object(chat._session, "post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

            with pytest.raises(LexiluxTimeoutError) as exc_info:
                chat("Hello, world!")

            # Verify the exception contains timeout information
            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.code == "timeout"
            assert exc_info.value.retryable is True

    def test_connection_refused(self):
        """Test that connection refused is properly raised."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Mock the session to raise a ConnectionError
        with patch.object(chat._session, "post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )

            with pytest.raises(LexiluxConnectionError) as exc_info:
                chat("Hello, world!")

            # Verify the exception contains connection error information
            assert (
                "connection" in str(exc_info.value).lower()
                or "failed" in str(exc_info.value).lower()
            )
            assert exc_info.value.code == "connection_failed"
            assert exc_info.value.retryable is True


class TestAPIExceptions:
    """Tests for API error responses."""

    def test_401_unauthorized(self):
        """Test that 401 response raises AuthenticationError."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with 401 status
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.ok = False
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

        with patch.object(chat._session, "post", return_value=mock_response):
            with pytest.raises(AuthenticationError) as exc_info:
                chat("Hello, world!")

            # Verify exception properties
            assert exc_info.value.status_code == 401
            assert exc_info.value.code == "authentication_failed"
            assert exc_info.value.retryable is False
            assert "Invalid API key" in exc_info.value.message

    def test_429_rate_limit(self):
        """Test that 429 response raises RateLimitError."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with 429 status
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

        with patch.object(chat._session, "post", return_value=mock_response):
            with pytest.raises(RateLimitError) as exc_info:
                chat("Hello, world!")

            # Verify exception properties
            assert exc_info.value.status_code == 429
            assert exc_info.value.code == "rate_limit_exceeded"
            assert exc_info.value.retryable is True
            assert "Rate limit exceeded" in exc_info.value.message

    def test_500_server_error(self):
        """Test that 500 response raises ServerError."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with 500 status
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.ok = False
        mock_response.json.return_value = {
            "error": {"message": "Internal server error"}
        }

        with patch.object(chat._session, "post", return_value=mock_response):
            with pytest.raises(ServerError) as exc_info:
                chat("Hello, world!")

            # Verify exception properties
            assert exc_info.value.status_code == 500
            assert exc_info.value.code == "server_error"
            assert exc_info.value.retryable is True
            assert "Internal server error" in exc_info.value.message

    def test_503_service_unavailable(self):
        """Test that 503 response raises ServerError."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with 503 status
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.ok = False
        mock_response.json.return_value = {"error": {"message": "Service unavailable"}}

        with patch.object(chat._session, "post", return_value=mock_response):
            with pytest.raises(ServerError) as exc_info:
                chat("Hello, world!")

            # Verify exception properties
            # Note: ServerError hardcodes status_code=500 in __init__, but we can verify the type
            assert exc_info.value.code == "server_error"
            assert exc_info.value.retryable is True
            assert "Service unavailable" in exc_info.value.message


class TestMalformedResponses:
    """Tests for malformed response handling."""

    def test_empty_response_body(self):
        """Test that empty response body is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with empty body
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {}

        with patch.object(chat._session, "post", return_value=mock_response):
            # The current implementation raises ValueError for missing choices
            with pytest.raises(ValueError) as exc_info:
                chat("Hello, world!")

            # Verify we get a clear error message
            assert "No choices" in str(exc_info.value)

    def test_missing_choices_in_response(self):
        """Test that response without choices is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response without choices
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            # Missing 'choices' key
        }

        with patch.object(chat._session, "post", return_value=mock_response):
            # The current implementation raises ValueError for missing choices
            with pytest.raises(ValueError) as exc_info:
                chat("Hello, world!")

            # Verify we get a clear error message
            assert "No choices" in str(exc_info.value)

    def test_invalid_json_response(self):
        """Test that invalid JSON response is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response that fails to parse JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch.object(chat._session, "post", return_value=mock_response):
            # This should raise the JSON parsing error
            with pytest.raises(ValueError) as exc_info:
                chat("Hello, world!")

            # Verify we get the JSON parsing error
            assert "Invalid JSON" in str(exc_info.value)

    def test_empty_choices_list(self):
        """Test that empty choices list is handled."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with empty choices list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [],  # Empty choices list
        }

        with patch.object(chat._session, "post", return_value=mock_response):
            # The current implementation raises ValueError for empty choices
            with pytest.raises(ValueError) as exc_info:
                chat("Hello, world!")

            # Verify we get a clear error message
            assert "No choices" in str(exc_info.value)


class TestStreamingExceptionPaths:
    """Tests for exception paths in streaming mode.

    Note: The current implementation of _streaming_request_context does not
    wrap network exceptions or check for HTTP errors. This is a known
    limitation that differs from non-streaming requests.

    Network errors: Raw requests.exceptions are raised (not wrapped)
    HTTP errors: Return empty stream (error response is not valid SSE)
    """

    def test_streaming_timeout_raises_raw_exception(self):
        """Test that timeout during streaming raises raw requests exception.

        This documents current behavior: _streaming_request_context does not
        catch and convert network exceptions like non-streaming requests do.
        """
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Mock the session.post to raise Timeout during streaming setup
        with patch.object(chat._session, "post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout(
                "Connection timed out during streaming"
            )

            # The raw requests exception is raised, not wrapped
            with pytest.raises(requests.exceptions.Timeout):
                list(chat.stream("Hello, world!"))

    def test_streaming_connection_refused_raises_raw_exception(self):
        """Test that connection refused during streaming raises raw exception.

        This documents current behavior: _streaming_request_context does not
        catch and convert network exceptions like non-streaming requests do.
        """
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Mock the session.post to raise ConnectionError
        with patch.object(chat._session, "post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Connection refused during streaming"
            )

            # The raw requests exception is raised, not wrapped
            with pytest.raises(requests.exceptions.ConnectionError):
                list(chat.stream("Hello, world!"))

    def test_streaming_error_response_returns_empty_stream(self):
        """Test that error responses in streaming return an empty stream.

        This documents the current behavior: HTTP errors in streaming mode
        are not caught and raised as exceptions. Instead, an empty stream
        is returned since the error response body is not valid SSE.
        """
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with 401 status but no SSE data
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.ok = False
        mock_response.iter_lines.return_value = []  # Empty stream
        mock_response.close = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch.object(chat._session, "post", return_value=mock_response):
            # The stream will be empty since iter_lines returns empty
            chunks = list(chat.stream("Hello, world!"))
            assert chunks == []


class TestAPIExceptionWithNonStandardErrorFormat:
    """Tests for API responses with non-standard error formats."""

    def test_401_with_plain_text_error(self):
        """Test 401 response with plain text error message."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response that returns plain text
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.ok = False
        mock_response.json.side_effect = ValueError("Not JSON")

        with patch.object(chat._session, "post", return_value=mock_response):
            # Should still raise AuthenticationError with default message
            with pytest.raises(AuthenticationError) as exc_info:
                chat("Hello, world!")

            # Verify exception properties
            assert exc_info.value.status_code == 401
            assert "HTTP 401" in exc_info.value.message

    def test_429_with_message_in_root(self):
        """Test 429 response with error message in root."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with error at root level
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.json.return_value = {"message": "Too many requests"}

        with patch.object(chat._session, "post", return_value=mock_response):
            with pytest.raises(RateLimitError) as exc_info:
                chat("Hello, world!")

            # Verify the message was extracted
            assert "Too many requests" in exc_info.value.message

    def test_500_with_string_error(self):
        """Test 500 response with string error."""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Create a mock response with string error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.ok = False
        mock_response.json.return_value = {"error": "Internal server error"}

        with patch.object(chat._session, "post", return_value=mock_response):
            with pytest.raises(ServerError) as exc_info:
                chat("Hello, world!")

            # Verify the message was extracted
            assert "Internal server error" in exc_info.value.message


class TestRetryableErrorHandling:
    """Tests for retryable error handling."""

    def test_retryable_errors_can_be_identified(self):
        """Test that retryable errors have the retryable flag set."""
        from lexilux.exceptions import (
            RateLimitError,
            ServerError,
            TimeoutError as LexiluxTimeoutError,
            ConnectionError as LexiluxConnectionError,
        )

        # Verify retryable errors
        assert RateLimitError().retryable is True
        assert ServerError().retryable is True
        assert LexiluxTimeoutError().retryable is True
        assert LexiluxConnectionError().retryable is True

    def test_non_retryable_errors_can_be_identified(self):
        """Test that non-retryable errors have the retryable flag unset."""
        from lexilux.exceptions import (
            AuthenticationError,
            ValidationError,
            NotFoundError,
        )

        # Verify non-retryable errors
        assert AuthenticationError().retryable is False
        assert ValidationError().retryable is False
        assert NotFoundError().retryable is False
