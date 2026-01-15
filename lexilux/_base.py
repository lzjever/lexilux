"""
Base HTTP client for all Lexilux API clients.

Provides common functionality:
- Session management with connection pooling
- Retry logic for failed requests
- Configurable timeouts
- Authentication handling
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from typing import TYPE_CHECKING, Any
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    pass


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
        ...     pool_connections=10,
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
        pool_connections: int = 10,
        pool_maxsize: int = 10,
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
            pool_connections: Number of connection pools to cache.
            pool_maxsize: Maximum number of connections in pool.
            headers: Additional headers to include in all requests.
            proxies: Proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.proxies = proxies

        # Configure timeout
        if connect_timeout_s is not None and read_timeout_s is not None:
            self.timeout = (connect_timeout_s, read_timeout_s)
        else:
            self.timeout = timeout_s

        # Create session with connection pooling
        self.session = self._create_session(
            max_retries=max_retries,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )

        # Prepare headers
        self.headers = self._prepare_headers(headers, api_key)

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
            requests.RequestException: On network or HTTP errors.
        """
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout,
            headers=self.headers,
            proxies=self.proxies,
        )
        response.raise_for_status()
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
            requests.RequestException: On network or HTTP errors.
        """
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout,
            headers=self.headers,
            proxies=self.proxies,
            stream=True,
        )
        response.raise_for_status()
        return response

    def close(self):
        """Close the session and release resources."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
