"""
Base HTTP client for all Lexilux API clients.

Provides common functionality:
- Connection pooling for improved performance
- Retry logic for failed requests
- Configurable timeouts
- Authentication handling
- Unified error handling
- Logging for debugging and monitoring
- Async support via httpx
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator
from urllib.parse import parse_qs, urlparse

import httpx
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

from lexilux.exceptions import (
    APIError,
    AuthenticationError,
    LexiluxError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from lexilux.exceptions import (
    ConnectionError as LexiluxConnectionError,
)
from lexilux.exceptions import (
    TimeoutError as LexiluxTimeoutError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BaseAPIClient:
    """
    Base API client with connection pooling for improved performance.

    All API clients (Chat, Embed, Rerank) should inherit from this class
    to get consistent HTTP behavior and configuration.

    Connection pooling is enabled by default to improve performance by reusing
    connections across multiple requests to the same host.

    Attributes:
        base_url: Base URL for API requests (without trailing slash).
        api_key: API key for authentication (optional).
        timeout: Request timeout in seconds (float or tuple for connect/read).
        headers: Default headers for all requests.
        proxies: Proxy configuration (None means use environment variables).
        pool_size: Connection pool size for HTTP adapter.

    Examples:
        >>> client = BaseAPIClient(
        ...     base_url="https://api.example.com/v1",
        ...     api_key="sk-...",
        ...     connect_timeout_s=5,
        ...     read_timeout_s=30,
        ...     max_retries=2,
        ... )
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        max_retries: int = 0,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
        pool_size: int = 2,
    ):
        """
        Initialize base API client.

        Args:
            base_url: Base URL for API requests (e.g., "https://api.openai.com/v1").
            api_key: API key for authentication (added to Authorization header).
            timeout_s: Default timeout for both connect and read (in seconds).
            connect_timeout_s: Connection timeout (overrides timeout_s if both set).
            read_timeout_s: Read timeout (overrides timeout_s if both set).
            max_retries: Maximum number of retries for failed requests (0 = disable).
            headers: Additional headers to include in all requests.
            proxies: Proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.
            pool_size: Connection pool size for HTTP adapter (default: 2).
        """
        if pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {pool_size}")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.proxies = proxies
        self._max_retries = max_retries

        # Configure timeout
        if connect_timeout_s is not None and read_timeout_s is not None:
            self.timeout = (connect_timeout_s, read_timeout_s)
        else:
            self.timeout = timeout_s

        # Async client (lazy initialization)
        self._async_client: httpx.AsyncClient | None = None

        # Prepare headers
        self.headers = self._prepare_headers(headers, api_key)

        # Create Session and configure connection pool
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _prepare_headers(
        self,
        headers: dict[str, str] | None,
        api_key: str | None,
    ) -> dict[str, str]:
        """
        Prepare request headers with authentication.

        Args:
            headers: Additional headers.
            api_key: API key for Bearer authentication.

        Returns:
            Headers dict with authentication and default headers.
        """
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")

        if api_key:
            headers.setdefault("Authorization", f"Bearer {api_key}")

        return headers

    def _sanitize_for_logging(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, str] | None]:
        """
        脱敏 URL 和 headers 中的敏感信息。

        Returns:
            (sanitized_url, sanitized_headers)
        """
        # URL 脱敏
        parsed = urlparse(url)
        sensitive_params = {"api_key", "token", "password"}

        # Parse and sanitize query parameters
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        sanitized_query_parts = []

        for key, values in query_params.items():
            if key.lower() in sensitive_params:
                # For sensitive params, use *** without encoding
                sanitized_query_parts.append(f"{key}=***")
            else:
                # For non-sensitive params, preserve original encoding
                # We need to manually reconstruct from the original query string
                # since parse_qs decodes everything
                for value in values:
                    sanitized_query_parts.append(f"{key}={value}")

        sanitized_query = "&".join(sanitized_query_parts)
        sanitized_url = parsed._replace(query=sanitized_query).geturl()

        # Headers 脱敏
        if headers:
            sensitive_headers = {
                "authorization",
                "cookie",
                "set-cookie",
                "x-api-key",
                "x-auth-token",
            }
            sanitized_headers = {
                k: "***" if k.lower() in sensitive_headers else v
                for k, v in headers.items()
            }
        else:
            sanitized_headers = None

        return sanitized_url, sanitized_headers

    def _map_exception(self, exc: requests.exceptions.RequestException) -> LexiluxError:
        """将 requests 异常映射到 Lexilux 异常"""
        if isinstance(exc, requests.exceptions.Timeout):
            return LexiluxTimeoutError(str(exc))
        elif isinstance(exc, requests.exceptions.ConnectionError):
            return LexiluxConnectionError(str(exc))
        else:
            return NetworkError(str(exc))

    def _get_retry_decorator(self, max_attempts: int):
        """
        获取重试装饰器

        Args:
            max_attempts: 最大尝试次数（包括首次请求）

        Returns:
            重试装饰器或空装饰器（如果 max_attempts <= 1）
        """
        if max_attempts <= 1:
            return lambda f: f  # 不重试

        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=60)
            + wait_random(0, 0.1),
            retry=retry_if_exception(
                lambda e: isinstance(e, LexiluxError) and e.retryable
            ),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
            reraise=True,
        )

    def _do_request(self, endpoint: str, payload: dict) -> requests.Response:
        """
        执行请求（可被重试）

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Returns:
            requests.Response object.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()

        # Sanitize URL and headers for logging
        sanitized_url, _ = self._sanitize_for_logging(url, self.headers)

        logger.debug("Making POST request to %s", sanitized_url)
        logger.debug("Request timeout: %s", self.timeout)

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers=self.headers,
                proxies=self.proxies,
            )
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            logger.error("Request timeout after %.2fs: %s", elapsed, sanitized_url)
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            logger.error("Connection failed after %.2fs: %s", elapsed, sanitized_url)
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            logger.error(
                "Request failed after %.2fs: %s - %s", elapsed, sanitized_url, e
            )
            # Generic requests error
            raise APIError(f"Request failed: {e}") from e

        elapsed = time.time() - start_time

        # Handle HTTP error status codes
        if not response.ok:
            logger.warning(
                "Request failed with status %d after %.2fs: %s",
                response.status_code,
                elapsed,
                sanitized_url,
            )
            self._handle_response_error(response)

        logger.info(
            "Request completed in %.2fs with status %d: %s",
            elapsed,
            response.status_code,
            sanitized_url,
        )

        return response

    def _handle_response_error(self, response: requests.Response) -> None:
        """
        Handle HTTP error responses and raise appropriate Lexilux exceptions.

        Args:
            response: The error response from the API.

        Raises:
            AuthenticationError: For 401 status codes.
            RateLimitError: For 429 status codes.
            NotFoundError: For 404 status codes.
            InvalidRequestError: For 400 status codes.
            ServerError: For 5xx status codes.
            APIError: For other error status codes.
        """
        status_code = response.status_code

        # Try to extract error message from response body
        error_message = f"HTTP {status_code}"
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                # OpenAI-style error
                if "error" in error_data:
                    error_info = error_data["error"]
                    if isinstance(error_info, dict):
                        error_message = error_info.get("message", error_message)
                    else:
                        error_message = str(error_info)
                else:
                    error_message = error_data.get("message", error_message)
        except (ValueError, KeyError):
            # Not JSON or no error field, use default message
            pass

        # Map status codes to specific exceptions
        if status_code == 401:
            raise AuthenticationError(error_message)
        elif status_code == 429:
            raise RateLimitError(error_message)
        elif status_code == 404:
            raise NotFoundError(error_message)
        elif status_code == 400:
            raise ValidationError(error_message)
        elif 500 <= status_code < 600:
            raise ServerError(error_message)
        else:
            raise APIError(
                message=error_message,
                status_code=status_code,
                code="http_error",
                retryable=False,
            )

    def _make_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        """
        Send POST request to API endpoint using connection pool.

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Returns:
            requests.Response object.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        # 获取重试装饰器并应用到请求方法
        retry_decorator = self._get_retry_decorator(self._max_retries + 1)
        request_func = retry_decorator(self._do_request)

        try:
            response = request_func(endpoint, payload)
        except requests.exceptions.RequestException as e:
            # Map any remaining request exceptions to Lexilux errors
            if isinstance(e, requests.exceptions.Timeout):
                raise LexiluxTimeoutError(f"Request timeout: {e}") from e
            elif isinstance(e, requests.exceptions.ConnectionError):
                raise LexiluxConnectionError(f"Connection failed: {e}") from e
            else:
                raise APIError(f"Request failed: {e}") from e

        # Close the response to return connection to pool
        response.close()

        return response

    def _make_streaming_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        """
        Send streaming POST request to API endpoint using connection pool.

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Returns:
            requests.Response object with stream=True.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()

        # Sanitize URL and headers for logging
        sanitized_url, _ = self._sanitize_for_logging(url, self.headers)

        logger.debug("Making streaming POST request to %s", sanitized_url)
        logger.debug("Request timeout: %s", self.timeout)

        # Use session for connection pooling
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers=self.headers,
                proxies=self.proxies,
                stream=True,
            )
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            logger.error(
                "Streaming request timeout after %.2fs: %s", elapsed, sanitized_url
            )
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Streaming connection failed after %.2fs: %s", elapsed, sanitized_url
            )
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            logger.error(
                "Streaming request failed after %.2fs: %s - %s",
                elapsed,
                sanitized_url,
                e,
            )
            # Generic requests error
            raise APIError(f"Request failed: {e}") from e

        elapsed = time.time() - start_time

        # Handle HTTP error status codes
        if not response.ok:
            logger.warning(
                "Streaming request failed with status %d after %.2fs: %s",
                response.status_code,
                elapsed,
                sanitized_url,
            )
            self._handle_response_error(response)
            # Close the response on error
            response.close()

        logger.info(
            "Streaming request initiated in %.2fs with status %d: %s",
            elapsed,
            response.status_code,
            sanitized_url,
        )

        # Note: Caller is responsible for closing the response when done streaming
        # to return connection to pool
        return response

    # =========================================================================
    # Async Methods (using httpx)
    # =========================================================================

    def _get_async_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client (lazy initialization).

        Returns:
            httpx.AsyncClient instance (no connection pooling).
        """
        if self._async_client is None:
            # Configure timeout
            if isinstance(self.timeout, tuple):
                timeout = httpx.Timeout(
                    connect=self.timeout[0],
                    read=self.timeout[1],
                    write=self.timeout[1],
                    pool=self.timeout[0],
                )
            else:
                timeout = httpx.Timeout(self.timeout)

            # Create async client without connection limits
            self._async_client = httpx.AsyncClient(
                timeout=timeout,
                headers=self.headers,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            )

        return self._async_client

    def _handle_async_response_error(self, response: httpx.Response) -> None:
        """
        Handle HTTP error responses from httpx and raise appropriate Lexilux exceptions.

        Args:
            response: The error response from the API.

        Raises:
            AuthenticationError: For 401 status codes.
            RateLimitError: For 429 status codes.
            NotFoundError: For 404 status codes.
            ValidationError: For 400 status codes.
            ServerError: For 5xx status codes.
            APIError: For other error status codes.
        """
        status_code = response.status_code

        # Try to extract error message from response body
        error_message = f"HTTP {status_code}"
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                # OpenAI-style error
                if "error" in error_data:
                    error_info = error_data["error"]
                    if isinstance(error_info, dict):
                        error_message = error_info.get("message", error_message)
                    else:
                        error_message = str(error_info)
                else:
                    error_message = error_data.get("message", error_message)
        except (ValueError, KeyError):
            # Not JSON or no error field, use default message
            pass

        # Map status codes to specific exceptions
        if status_code == 401:
            raise AuthenticationError(error_message)
        elif status_code == 429:
            raise RateLimitError(error_message)
        elif status_code == 404:
            raise NotFoundError(error_message)
        elif status_code == 400:
            raise ValidationError(error_message)
        elif 500 <= status_code < 600:
            raise ServerError(error_message)
        else:
            raise APIError(
                message=error_message,
                status_code=status_code,
                code="http_error",
                retryable=False,
            )

    async def _ado_request(self, endpoint: str, payload: dict) -> httpx.Response:
        """
        执行异步请求（可被重试）

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Returns:
            httpx.Response object.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()

        # Sanitize URL and headers for logging
        sanitized_url, _ = self._sanitize_for_logging(url, self.headers)

        logger.debug("Making async POST request to %s", sanitized_url)

        client = self._get_async_client()

        try:
            response = await client.post(
                url,
                json=payload,
            )
        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async request timeout after %.2fs: %s", elapsed, sanitized_url
            )
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except httpx.ConnectError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async connection failed after %.2fs: %s", elapsed, sanitized_url
            )
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async request failed after %.2fs: %s - %s", elapsed, sanitized_url, e
            )
            raise APIError(f"Request failed: {e}") from e

        elapsed = time.time() - start_time

        # Handle HTTP error status codes
        if not response.is_success:
            logger.warning(
                "Async request failed with status %d after %.2fs: %s",
                response.status_code,
                elapsed,
                sanitized_url,
            )
            self._handle_async_response_error(response)

        logger.info(
            "Async request completed in %.2fs with status %d: %s",
            elapsed,
            response.status_code,
            sanitized_url,
        )

        return response

    async def _amake_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        """
        Send async POST request to API endpoint.

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Returns:
            httpx.Response object.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        # 获取重试装饰器并应用到异步请求方法
        retry_decorator = self._get_retry_decorator(self._max_retries + 1)
        request_func = retry_decorator(self._ado_request)

        try:
            response = await request_func(endpoint, payload)
        except httpx.HTTPError as e:
            # Map any remaining httpx exceptions to Lexilux errors
            if isinstance(e, httpx.TimeoutException):
                raise LexiluxTimeoutError(f"Request timeout: {e}") from e
            elif isinstance(e, httpx.ConnectError):
                raise LexiluxConnectionError(f"Connection failed: {e}") from e
            else:
                raise APIError(f"Request failed: {e}") from e

        return response

    async def _amake_streaming_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Send async streaming POST request to API endpoint.

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Yields:
            Lines from the SSE stream.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()

        # Sanitize URL and headers for logging
        sanitized_url, _ = self._sanitize_for_logging(url, self.headers)

        logger.debug("Making async streaming POST request to %s", sanitized_url)

        client = self._get_async_client()

        try:
            async with client.stream("POST", url, json=payload) as response:
                elapsed = time.time() - start_time

                # Handle HTTP error status codes
                if not response.is_success:
                    # Read the body to get error message
                    await response.aread()
                    logger.warning(
                        "Async streaming request failed with status %d after %.2fs: %s",
                        response.status_code,
                        elapsed,
                        sanitized_url,
                    )
                    self._handle_async_response_error(response)

                logger.info(
                    "Async streaming request initiated in %.2fs with status %d: %s",
                    elapsed,
                    response.status_code,
                    sanitized_url,
                )

                # Yield lines from the stream
                async for line in response.aiter_lines():
                    if line:
                        yield line

        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async streaming request timeout after %.2fs: %s",
                elapsed,
                sanitized_url,
            )
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except httpx.ConnectError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async streaming connection failed after %.2fs: %s",
                elapsed,
                sanitized_url,
            )
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async streaming request failed after %.2fs: %s - %s",
                elapsed,
                sanitized_url,
                e,
            )
            raise APIError(f"Request failed: {e}") from e

    async def aclose(self) -> None:
        """Close the async client and release resources."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def close(self):
        """Close the session and release resources."""
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
