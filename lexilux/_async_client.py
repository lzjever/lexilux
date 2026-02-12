"""
Async client mixin for HTTP clients.

Provides common async client management, context manager support,
and cleanup functionality for API clients that don't need the full
BaseAPIClient features (connection pooling, retries, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass


class AsyncClientMixin:
    """
    Mixin providing async HTTP client management and context managers.

    This mixin provides:
    - Lazy async client initialization via `_get_async_client()`
    - Async cleanup via `aclose()`
    - Sync/async context manager support

    Classes using this mixin must set:
    - `self._async_client: httpx.AsyncClient | None` to None in __init__
    - `self.headers: dict[str, str]` for HTTP headers
    - `self.timeout_s: float` for timeout
    - `self.proxies: dict[str, str] | None` for proxy configuration

    Examples:
        >>> class MyClient(AsyncClientMixin):
        ...     def __init__(self, base_url: str, api_key: str, timeout_s: float = 60.0):
        ...         self.base_url = base_url
        ...         self.headers = {"Authorization": f"Bearer {api_key}"}
        ...         self.timeout_s = timeout_s
        ...         self.proxies = None
        ...         self._async_client = None
        ...
        ...     async def fetch(self):
        ...         client = self._get_async_client()
        ...         response = await client.get(f"{self.base_url}/data")
        ...         return response.json()
    """

    _async_client: httpx.AsyncClient | None
    headers: dict[str, str]
    timeout_s: float
    proxies: dict[str, str] | None

    def _get_async_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client.

        Returns:
            httpx.AsyncClient instance with configured timeout, headers, and proxies.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                headers=self.headers,
                proxies=self.proxies,
            )
        return self._async_client

    async def aclose(self) -> None:
        """
        Close the async client and release resources.

        Should be called when done with the client, or use async context manager.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def close(self) -> None:
        """
        Close sync resources.

        Placeholder for consistency with async clients. Override if sync cleanup needed.
        """
        pass

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.close()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
        return False
