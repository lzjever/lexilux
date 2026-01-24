"""
Base HTTP client for all Lexilux API clients.

Provides common functionality:
- Session management with connection pooling
- Retry logic for failed requests
- Configurable timeouts
- Authentication handling
- Unified error handling
- Logging for debugging and monitoring
- Async support via httpx
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lexilux.exceptions import (
    APIError,
    AuthenticationError,
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
    Base API client with connection pooling and retry support.

    All API clients (Chat, Embed, Rerank) should inherit from this class
    to get consistent HTTP behavior and configuration.

    Attributes:
        base_url: Base URL for API requests (without trailing slash).
        api_key: API key for authentication (optional).
        timeout: Request timeout in seconds (float or tuple for connect/read).
        session: requests.Session instance with connection pooling.
        headers: Default headers for all requests.
        proxies: Proxy configuration (None means use environment variables).

    Examples:
        >>> client = BaseAPIClient(
        ...     base_url="https://api.example.com/v1",
        ...     api_key="sk-...",
        ...     connect_timeout_s=5,
        ...     read_timeout_s=30,
        ...     max_retries=2,
        ...     pool_connections=1,    # One host (default)
        ...     pool_maxsize=1,        # One connection (safe default for most LLM APIs)
        ... )
        >>> # For concurrent usage:
        >>> concurrent_client = BaseAPIClient(
        ...     base_url="https://api.example.com/v1",
        ...     pool_maxsize=3,        # Allow 3 concurrent connections
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
        pool_connections: int = 1,
        pool_maxsize: int = 1,
        connection_idle_timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
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
            pool_connections: Number of different hosts to cache connection pools for (default: 1).
                                Since each client connects to one API endpoint, 1 is sufficient.
            pool_maxsize: Maximum connections per host pool (default: 1).
                         Conservative default for single-threaded usage and strict API limits.
                         Increase for concurrent requests: pool_maxsize=3 for APIs allowing 3 concurrent.
            connection_idle_timeout: Seconds to wait before closing idle connections after business completion (default: 30.0).
                                   Set to 0 to disable auto-cleanup.
            headers: Additional headers to include in all requests.
            proxies: Proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.proxies = proxies
        self.connection_idle_timeout = connection_idle_timeout

        # Configure timeout
        if connect_timeout_s is not None and read_timeout_s is not None:
            self.timeout = (connect_timeout_s, read_timeout_s)
        else:
            self.timeout = timeout_s

        # Store session config for lazy initialization
        self._session_config = {
            "max_retries": max_retries,
            "pool_connections": pool_connections,
            "pool_maxsize": pool_maxsize,
        }

        # Lazy initialization for both sync and async clients
        self._session: requests.Session | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._max_retries = max_retries

        # Connection management
        self._last_request_time = 0.0
        self._cleanup_timer: threading.Timer | None = None
        self._session_lock = threading.RLock()

        # Prepare headers
        self.headers = self._prepare_headers(headers, api_key)

    def _get_session(self) -> requests.Session:
        """
        Get or create the HTTP session (lazy initialization with auto-cleanup).

        Returns:
            requests.Session instance configured with connection pooling.
        """
        with self._session_lock:
            if self._session is None:
                logger.debug("Creating HTTP session (lazy initialization)")
                self._session = self._create_session(
                    max_retries=self._session_config["max_retries"],
                    pool_connections=self._session_config["pool_connections"],
                    pool_maxsize=self._session_config["pool_maxsize"],
                )

            # Cancel existing cleanup timer since we're using the session
            self._cancel_cleanup_timer()
            self._last_request_time = time.time()

            return self._session

    def _create_session(
        self,
        max_retries: int,
        pool_connections: int,
        pool_maxsize: int,
    ) -> requests.Session:
        """
        Create a requests.Session with connection pooling and retry.

        Args:
            max_retries: Maximum number of retries.
            pool_connections: Number of connection pools.
            pool_maxsize: Maximum pool size.

        Returns:
            Configured requests.Session instance.
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.1,  # Wait 0.1s, 0.2s, 0.4s... between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["POST", "GET", "PUT", "DELETE"],  # Retry for these methods
        )

        # Create adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy,
        )

        # Mount adapter for both http and https
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _schedule_connection_cleanup(self) -> None:
        """
        Schedule connection cleanup after business completion.
        Only schedules if idle timeout > 0.
        """
        if self.connection_idle_timeout <= 0:
            return

        with self._session_lock:
            # Cancel any existing timer
            self._cancel_cleanup_timer()

            # Schedule new cleanup
            self._cleanup_timer = threading.Timer(
                self.connection_idle_timeout, self._cleanup_connections
            )
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

            logger.debug(
                "Scheduled connection cleanup in %.1fs", self.connection_idle_timeout
            )

    def _cancel_cleanup_timer(self) -> None:
        """Cancel pending cleanup timer."""
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    def _cleanup_connections(self) -> None:
        """Clean up idle connections."""
        with self._session_lock:
            elapsed = time.time() - self._last_request_time

            # Double check - maybe there was a recent request
            if elapsed < self.connection_idle_timeout:
                logger.debug("Skipping cleanup - recent activity (%.1fs ago)", elapsed)
                return

            logger.debug("Cleaning up idle connections after %.1fs", elapsed)

            # Close sync session
            if self._session is not None:
                self._session.close()
                self._session = None
                logger.debug("Closed HTTP session")

            # Note: Async client cleanup is handled separately in aclose()

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
        Send POST request to API endpoint.

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

        logger.debug("Making POST request to %s", url)
        logger.debug("Request timeout: %s", self.timeout)

        session = self._get_session()  # Lazy initialization

        try:
            response = session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers=self.headers,
                proxies=self.proxies,
            )
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            logger.error("Request timeout after %.2fs: %s", elapsed, url)
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            logger.error("Connection failed after %.2fs: %s", elapsed, url)
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            logger.error("Request failed after %.2fs: %s - %s", elapsed, url, e)
            # Generic requests error
            raise APIError(f"Request failed: {e}") from e

        elapsed = time.time() - start_time

        # Handle HTTP error status codes
        if not response.ok:
            logger.warning(
                "Request failed with status %d after %.2fs: %s",
                response.status_code,
                elapsed,
                url,
            )
            self._handle_response_error(response)

        logger.info(
            "Request completed in %.2fs with status %d: %s",
            elapsed,
            response.status_code,
            url,
        )

        # Schedule connection cleanup after business completion
        self._schedule_connection_cleanup()

        return response

    def _make_streaming_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        """
        Send streaming POST request to API endpoint.

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

        logger.debug("Making streaming POST request to %s", url)
        logger.debug("Request timeout: %s", self.timeout)

        session = self._get_session()  # Lazy initialization

        try:
            response = session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers=self.headers,
                proxies=self.proxies,
                stream=True,
            )
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            logger.error("Streaming request timeout after %.2fs: %s", elapsed, url)
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            logger.error("Streaming connection failed after %.2fs: %s", elapsed, url)
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            logger.error(
                "Streaming request failed after %.2fs: %s - %s", elapsed, url, e
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
                url,
            )
            self._handle_response_error(response)

        logger.info(
            "Streaming request initiated in %.2fs with status %d: %s",
            elapsed,
            response.status_code,
            url,
        )

        # Note: For streaming, we'll schedule cleanup when the stream is consumed
        # This is handled at the business layer (e.g., when iterator finishes)

        return response

    # =========================================================================
    # Async Methods (using httpx)
    # =========================================================================

    def _get_async_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client (lazy initialization).

        Returns:
            httpx.AsyncClient instance configured with connection pooling and retry.
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

            # Configure transport with retries
            # httpx uses transport for connection pooling and retries
            transport = httpx.AsyncHTTPTransport(
                retries=self._max_retries,
            )

            # Create async client
            self._async_client = httpx.AsyncClient(
                timeout=timeout,
                headers=self.headers,
                transport=transport,
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
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()

        logger.debug("Making async POST request to %s", url)

        client = self._get_async_client()

        try:
            response = await client.post(
                url,
                json=payload,
            )
        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time
            logger.error("Async request timeout after %.2fs: %s", elapsed, url)
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except httpx.ConnectError as e:
            elapsed = time.time() - start_time
            logger.error("Async connection failed after %.2fs: %s", elapsed, url)
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPError as e:
            elapsed = time.time() - start_time
            logger.error("Async request failed after %.2fs: %s - %s", elapsed, url, e)
            raise APIError(f"Request failed: {e}") from e

        elapsed = time.time() - start_time

        # Handle HTTP error status codes
        if not response.is_success:
            logger.warning(
                "Async request failed with status %d after %.2fs: %s",
                response.status_code,
                elapsed,
                url,
            )
            self._handle_async_response_error(response)

        logger.info(
            "Async request completed in %.2fs with status %d: %s",
            elapsed,
            response.status_code,
            url,
        )
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

        logger.debug("Making async streaming POST request to %s", url)

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
                        url,
                    )
                    self._handle_async_response_error(response)

                logger.info(
                    "Async streaming request initiated in %.2fs with status %d: %s",
                    elapsed,
                    response.status_code,
                    url,
                )

                # Yield lines from the stream
                async for line in response.aiter_lines():
                    if line:
                        yield line

        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async streaming request timeout after %.2fs: %s", elapsed, url
            )
            raise LexiluxTimeoutError(f"Request timeout: {e}") from e
        except httpx.ConnectError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async streaming connection failed after %.2fs: %s", elapsed, url
            )
            raise LexiluxConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPError as e:
            elapsed = time.time() - start_time
            logger.error(
                "Async streaming request failed after %.2fs: %s - %s", elapsed, url, e
            )
            raise APIError(f"Request failed: {e}") from e

    async def aclose(self) -> None:
        """Close the async client and release resources."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def close(self):
        """Close the session and release resources."""
        with self._session_lock:
            # Cancel any pending cleanup timer
            self._cancel_cleanup_timer()

            # Close session if it exists
            if self._session is not None:
                self._session.close()
                self._session = None
                logger.debug("Closed HTTP session")

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
