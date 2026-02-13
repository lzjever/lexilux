"""Tests for BaseAPIClient connection pool functionality."""

import requests
import pytest
import unittest
from unittest.mock import Mock, patch

from lexilux._base import BaseAPIClient
from lexilux.exceptions import (
    RateLimitError,
    AuthenticationError,
    ServerError,
    NetworkError,
    TimeoutError as LexiluxTimeoutError,
    ConnectionError as LexiluxConnectionError,
)


def test_connection_pool_initialization():
    """Verify connection pool is correctly initialized"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        pool_size=5,
    )
    # Verify Session is created
    assert hasattr(client, "_session")
    assert isinstance(client._session, requests.Session)


def test_connection_pool_default_size():
    """Verify default connection pool size is 2"""
    client = BaseAPIClient(base_url="https://api.example.com")
    # Verify default pool_size=2
    adapter = client._session.get_adapter("https://api.example.com")
    assert hasattr(adapter, "_pool_connections")
    assert adapter._pool_connections == 2


def test_connection_pool_custom_size():
    """Verify custom connection pool size"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        pool_size=20,
    )
    adapter = client._session.get_adapter("https://api.example.com")
    assert adapter._pool_connections == 20
    assert adapter._pool_maxsize == 20


def test_pool_size_validation():
    """Verify pool_size must be >= 1"""
    with pytest.raises(ValueError, match="pool_size must be at least 1"):
        BaseAPIClient(base_url="https://api.example.com", pool_size=0)

    with pytest.raises(ValueError, match="pool_size must be at least 1"):
        BaseAPIClient(base_url="https://api.example.com", pool_size=-1)


def test_close_method():
    """Verify close() method properly closes Session"""
    client = BaseAPIClient(base_url="https://api.example.com")

    # Session should be open
    assert client._session is not None

    # Close the client
    client.close()

    # After close, the session object still exists but connections are closed
    # We can verify close() was called by checking that we can call it again
    # without error (idempotent)
    client.close()  # Should not raise


def test_context_manager():
    """Verify context manager properly closes resources"""
    client = BaseAPIClient(base_url="https://api.example.com")

    with client as ctx:
        # Session should be open inside context
        assert ctx._session is not None

    # After exiting context, close() was called
    # We can verify by checking that calling close() again doesn't raise
    client.close()  # Should not raise


def test_make_request_uses_session():
    """Verify _make_request uses session instead of direct requests.post"""
    client = BaseAPIClient(base_url="https://api.example.com", api_key="test")

    with patch.object(client._session, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_response

        client._make_request("test", {"data": "test"})

        # Verify session.post is used
        assert mock_post.called
        # Verify correct parameters are passed
        call_args = mock_post.call_args
        assert "test" in call_args[0][0]  # URL contains endpoint


def test_retry_on_rate_limit_error():
    """Verify RateLimitError triggers retry"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=2,
    )

    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RateLimitError("Rate limited")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        return mock_response

    with patch.object(client, "_do_request", side_effect=mock_request):
        client._make_request("test", {})

    # Verify retried 2 times (first failed, second succeeded)
    assert call_count == 2


def test_no_retry_on_authentication_error():
    """Verify AuthenticationError does not trigger retry"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=2,
    )

    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("Invalid key")

    with patch.object(client, "_do_request", side_effect=mock_request):
        with pytest.raises(AuthenticationError):
            client._make_request("test", {})

    # Verify no retry occurred
    assert call_count == 1


def test_retry_on_server_error():
    """Verify ServerError triggers retry"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=2,
    )

    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ServerError("Internal server error")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        return mock_response

    with patch.object(client, "_do_request", side_effect=mock_request):
        client._make_request("test", {})

    # Verify retried 2 times
    assert call_count == 2


def test_retry_on_timeout_error():
    """Verify TimeoutError triggers retry"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=2,
    )

    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LexiluxTimeoutError("Request timeout")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        return mock_response

    with patch.object(client, "_do_request", side_effect=mock_request):
        client._make_request("test", {})

    # Verify retried 2 times
    assert call_count == 2


def test_retry_on_connection_error():
    """Verify ConnectionError triggers retry"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=2,
    )

    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LexiluxConnectionError("Connection failed")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        return mock_response

    with patch.object(client, "_do_request", side_effect=mock_request):
        client._make_request("test", {})

    # Verify retried 2 times
    assert call_count == 2


def test_no_retry_when_max_retries_is_zero():
    """Verify no retry when max_retries=0"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=0,
    )

    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RateLimitError("Rate limited")

    with patch.object(client, "_do_request", side_effect=mock_request):
        with pytest.raises(RateLimitError):
            client._make_request("test", {})

    # Verify only tried 1 time (no retry)
    assert call_count == 1


def test_map_timeout_exception():
    """Verify requests.Timeout maps to Lexilux TimeoutError"""
    client = BaseAPIClient(base_url="https://api.example.com")

    original_exc = requests.exceptions.Timeout("Connection timed out")
    mapped = client._map_exception(original_exc)

    assert isinstance(mapped, LexiluxTimeoutError)
    assert "timed out" in mapped.message.lower()


def test_map_connection_exception():
    """Verify requests.ConnectionError maps to Lexilux ConnectionError"""
    client = BaseAPIClient(base_url="https://api.example.com")

    original_exc = requests.exceptions.ConnectionError("Failed to connect")
    mapped = client._map_exception(original_exc)

    assert isinstance(mapped, LexiluxConnectionError)


def test_map_generic_request_exception():
    """Verify generic RequestException maps to NetworkError"""
    client = BaseAPIClient(base_url="https://api.example.com")

    original_exc = requests.exceptions.RequestException("Generic error")
    mapped = client._map_exception(original_exc)

    assert isinstance(mapped, NetworkError)


@pytest.mark.asyncio
async def test_async_retry_on_rate_limit_error():
    """Verify async methods also support retry"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        api_key="test",
        max_retries=2,
    )

    call_count = 0

    async def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RateLimitError("Rate limited")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"result": "ok"}
        return mock_response

    # Patch the instance method directly for async compatibility
    original_method = client._ado_request
    client._ado_request = mock_request

    try:
        await client._amake_request("test", {})
    finally:
        client._ado_request = original_method

    assert call_count == 2


def test_sanitize_url_with_api_key():
    """Verify api_key parameter in URL is sanitized"""
    client = BaseAPIClient(base_url="https://api.example.com")

    url = "https://api.example.com/v1/chat?api_key=sk-abc123&other=value"
    sanitized, _ = client._sanitize_for_logging(url)

    assert "api_key=***" in sanitized
    assert "sk-abc123" not in sanitized
    assert "other=value" in sanitized


def test_sanitize_headers_with_authorization():
    """Verify Authorization header is sanitized"""
    client = BaseAPIClient(base_url="https://api.example.com")

    headers = {
        "Authorization": "Bearer sk-abc123",
        "Content-Type": "application/json",
    }
    _, sanitized = client._sanitize_for_logging("", headers)

    assert sanitized["Authorization"] == "***"
    assert sanitized["Content-Type"] == "application/json"


def test_sanitize_headers_multiple_sensitive():
    """Verify all sensitive headers are sanitized"""
    client = BaseAPIClient(base_url="https://api.example.com")

    headers = {
        "Authorization": "Bearer token",
        "Cookie": "session=secret",
        "X-API-Key": "secret-key",
        "User-Agent": "Lexilux/2.5.0",
    }
    _, sanitized = client._sanitize_for_logging("", headers)

    assert sanitized["Authorization"] == "***"
    assert sanitized["Cookie"] == "***"
    assert sanitized["X-API-Key"] == "***"
    assert sanitized["User-Agent"] == "Lexilux/2.5.0"


def test_streaming_request_context_closes_response():
    """Verify that streaming request context closes response."""
    client = BaseAPIClient(base_url="https://api.example.com", api_key="test")

    # Mock the session.post to return a response with close tracking
    class MockResponse:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock_response = MockResponse()

    with unittest.mock.patch.object(
        client._session, "post", return_value=mock_response
    ):
        with client._streaming_request_context("test", {}):
            pass

    assert mock_response.closed, "Response should be closed after context exit"


def test_streaming_request_context_closes_on_exception():
    """Verify that streaming request context closes response even on exception."""
    client = BaseAPIClient(base_url="https://api.example.com", api_key="test")

    class MockResponse:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    mock_response = MockResponse()

    with unittest.mock.patch.object(
        client._session, "post", return_value=mock_response
    ):
        with pytest.raises(ValueError):
            with client._streaming_request_context("test", {}):
                raise ValueError("Test exception")

    assert mock_response.closed, "Response should be closed even on exception"


def test_ssl_verification_default():
    """Verify SSL verification is enabled by default"""
    client = BaseAPIClient(base_url="https://api.example.com")

    # Verify default verify_ssl=True
    assert client._verify_ssl is True


def test_ssl_verification_disabled():
    """Verify SSL verification can be disabled"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        verify_ssl=False,
    )

    # Verify verify_ssl=False
    assert client._verify_ssl is False


def test_ssl_verification_custom_ca_bundle():
    """Verify custom CA certificate can be used"""
    ca_bundle_path = "/path/to/ca.crt"
    client = BaseAPIClient(
        base_url="https://api.example.com",
        verify_ssl=ca_bundle_path,
    )

    # Verify verify_ssl is set to custom path
    assert client._verify_ssl == ca_bundle_path


def test_ssl_verification_used_in_request():
    """Verify verify_ssl parameter is passed to requests"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        verify_ssl="/custom/ca.crt",
        api_key="test",
    )

    with patch.object(client._session, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_response

        client._make_request("test", {"data": "test"})

        # Verify verify parameter is passed
        call_kwargs = mock_post.call_args[1]
        assert "verify" in call_kwargs
        assert call_kwargs["verify"] == "/custom/ca.crt"


def test_ssl_verification_disabled_passed_to_request():
    """Verify verify_ssl=False 被正确传递到 requests"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        verify_ssl=False,
        api_key="test",
    )

    with patch.object(client._session, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_response

        client._make_request("test", {"data": "test"})

        # Verify verify=False is passed
        call_kwargs = mock_post.call_args[1]
        assert "verify" in call_kwargs
        assert call_kwargs["verify"] is False


@pytest.mark.asyncio
async def test_ssl_verification_used_in_async_request():
    """验证 verify_ssl 参数被存储并可用于异步客户端"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        verify_ssl="/custom/ca.crt",
        api_key="test",
    )

    # Verify that verify_ssl is stored correctly
    assert client._verify_ssl == "/custom/ca.crt"
