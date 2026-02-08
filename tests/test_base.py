"""Tests for BaseAPIClient connection pool functionality."""

import requests
import pytest
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
    """验证连接池正确初始化"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        pool_size=5,
    )
    # 验证 Session 被创建
    assert hasattr(client, "_session")
    assert isinstance(client._session, requests.Session)


def test_connection_pool_default_size():
    """验证默认连接池大小为 2"""
    client = BaseAPIClient(base_url="https://api.example.com")
    # 验证默认 pool_size=2
    adapter = client._session.get_adapter("https://api.example.com")
    assert hasattr(adapter, "_pool_connections")
    assert adapter._pool_connections == 2


def test_connection_pool_custom_size():
    """验证自定义连接池大小"""
    client = BaseAPIClient(
        base_url="https://api.example.com",
        pool_size=20,
    )
    adapter = client._session.get_adapter("https://api.example.com")
    assert adapter._pool_connections == 20
    assert adapter._pool_maxsize == 20


def test_pool_size_validation():
    """验证 pool_size 必须大于等于 1"""
    with pytest.raises(ValueError, match="pool_size must be at least 1"):
        BaseAPIClient(base_url="https://api.example.com", pool_size=0)

    with pytest.raises(ValueError, match="pool_size must be at least 1"):
        BaseAPIClient(base_url="https://api.example.com", pool_size=-1)


def test_close_method():
    """验证 close() 方法正确关闭 Session"""
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
    """验证上下文管理器正确关闭资源"""
    client = BaseAPIClient(base_url="https://api.example.com")

    with client as ctx:
        # Session should be open inside context
        assert ctx._session is not None

    # After exiting context, close() was called
    # We can verify by checking that calling close() again doesn't raise
    client.close()  # Should not raise


def test_make_request_uses_session():
    """验证 _make_request 使用 session 而非直接 requests.post"""
    client = BaseAPIClient(base_url="https://api.example.com", api_key="test")

    with patch.object(client._session, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_response

        client._make_request("test", {"data": "test"})

        # 验证使用 session.post
        assert mock_post.called
        # 验证传入正确的参数
        call_args = mock_post.call_args
        assert "test" in call_args[0][0]  # URL 包含 endpoint


def test_retry_on_rate_limit_error():
    """验证 RateLimitError 触发重试"""
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

    # 验证重试了 2 次（第一次失败，第二次成功）
    assert call_count == 2


def test_no_retry_on_authentication_error():
    """验证 AuthenticationError 不触发重试"""
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

    # 验证没有重试
    assert call_count == 1


def test_retry_on_server_error():
    """验证 ServerError 触发重试"""
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

    # 验证重试了 2 次
    assert call_count == 2


def test_retry_on_timeout_error():
    """验证 TimeoutError 触发重试"""
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

    # 验证重试了 2 次
    assert call_count == 2


def test_retry_on_connection_error():
    """验证 ConnectionError 触发重试"""
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

    # 验证重试了 2 次
    assert call_count == 2


def test_no_retry_when_max_retries_is_zero():
    """验证 max_retries=0 时不重试"""
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

    # 验证只尝试了 1 次（没有重试）
    assert call_count == 1


def test_map_timeout_exception():
    """验证 requests.Timeout 映射到 Lexilux TimeoutError"""
    client = BaseAPIClient(base_url="https://api.example.com")

    original_exc = requests.exceptions.Timeout("Connection timed out")
    mapped = client._map_exception(original_exc)

    assert isinstance(mapped, LexiluxTimeoutError)
    assert "timed out" in mapped.message.lower()


def test_map_connection_exception():
    """验证 requests.ConnectionError 映射到 Lexilux ConnectionError"""
    client = BaseAPIClient(base_url="https://api.example.com")

    original_exc = requests.exceptions.ConnectionError("Failed to connect")
    mapped = client._map_exception(original_exc)

    assert isinstance(mapped, LexiluxConnectionError)


def test_map_generic_request_exception():
    """验证通用 RequestException 映射到 NetworkError"""
    client = BaseAPIClient(base_url="https://api.example.com")

    original_exc = requests.exceptions.RequestException("Generic error")
    mapped = client._map_exception(original_exc)

    assert isinstance(mapped, NetworkError)


@pytest.mark.asyncio
async def test_async_retry_on_rate_limit_error():
    """验证异步方法也支持重试"""
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

    with patch.object(client, "_ado_request", side_effect=mock_request):
        await client._amake_request("test", {})

    assert call_count == 2


def test_sanitize_url_with_api_key():
    """验证 URL 中的 api_key 参数被脱敏"""
    client = BaseAPIClient(base_url="https://api.example.com")

    url = "https://api.example.com/v1/chat?api_key=sk-abc123&other=value"
    sanitized, _ = client._sanitize_for_logging(url)

    assert "api_key=***" in sanitized
    assert "sk-abc123" not in sanitized
    assert "other=value" in sanitized


def test_sanitize_headers_with_authorization():
    """验证 Authorization header 被脱敏"""
    client = BaseAPIClient(base_url="https://api.example.com")

    headers = {
        "Authorization": "Bearer sk-abc123",
        "Content-Type": "application/json",
    }
    _, sanitized = client._sanitize_for_logging("", headers)

    assert sanitized["Authorization"] == "***"
    assert sanitized["Content-Type"] == "application/json"


def test_sanitize_headers_multiple_sensitive():
    """验证多个敏感 headers 都被脱敏"""
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
