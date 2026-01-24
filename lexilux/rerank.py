"""
Rerank API client.

Provides a simple, function-like API for document reranking with unified usage tracking.
Supports multiple provider modes: OpenAI-compatible and DashScope.
Supports both sync and async operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx
import requests

from lexilux.usage import Json, ResultBase, Usage
from lexilux.chat.utils import parse_usage

if TYPE_CHECKING:
    pass

# Type aliases
Ranked = list[tuple[int, float]]  # (index, score)
RankedWithDoc = list[tuple[int, float, str]]  # (index, score, doc)


class RerankResult(ResultBase):
    """
    Rerank result.

    The results field contains:
    - Ranked (List[Tuple[int, float]]) when include_docs=False
    - RankedWithDoc (List[Tuple[int, float, str]]) when include_docs=True

    Results are sorted by score in descending order.

    Attributes:
        results: Ranked results (with or without documents).
        usage: Usage statistics.
        raw: Raw API response.

    Examples:
        >>> result = rerank("python http", ["urllib", "requests", "httpx"])
        >>> ranked = result.results  # List[Tuple[int, float]]
        >>> print(ranked[0])  # (1, 0.95) - (index, score)

        >>> result = rerank("python http", ["urllib", "requests"], include_docs=True)
        >>> ranked = result.results  # List[Tuple[int, float, str]]
        >>> print(ranked[0])  # (1, 0.95, "requests") - (index, score, doc)
    """

    def __init__(
        self,
        *,
        results: Ranked | RankedWithDoc,
        usage: Usage,
        raw: Json | None = None,
    ):
        """
        Initialize RerankResult.

        Args:
            results: Ranked results.
            usage: Usage statistics.
            raw: Raw API response.
        """
        super().__init__(usage=usage, raw=raw)
        self.results = results

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RerankResult(results=[{len(self.results)} items], usage={self.usage!r})"
        )


class RerankModeHandler(ABC):
    """
    Abstract base class for rerank mode handlers.

    Each handler implements provider-specific request/response format conversion
    while maintaining a unified interface.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        headers: dict[str, str],
        timeout_s: float,
        proxies: dict[str, str] | None = None,
    ):
        """
        Initialize mode handler.

        Args:
            base_url: Base URL for the API.
            api_key: API key for authentication.
            headers: HTTP headers.
            timeout_s: Request timeout in seconds.
            proxies: Optional proxy configuration dict.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = headers
        self.timeout_s = timeout_s
        self.proxies = proxies

    @abstractmethod
    def build_request(
        self,
        query: str,
        docs: Sequence[str],
        model: str,
        top_k: int | None,
        include_docs: bool,
        extra: Json | None,
    ) -> tuple[str, Json]:
        """
        Build HTTP request for this mode.

        Args:
            query: Query string.
            docs: Document strings to rerank.
            model: Model identifier.
            top_k: Number of top results to return.
            include_docs: Whether to include documents in response.
            extra: Additional parameters.

        Returns:
            Tuple of (url, payload).
        """
        pass

    @abstractmethod
    def parse_response(
        self,
        response_data: Json,
        docs: Sequence[str],
        include_docs: bool,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float, str | None]], Usage]:
        """
        Parse API response for this mode.

        Args:
            response_data: Raw API response JSON.
            docs: Original document list (for index mapping).
            include_docs: Whether documents were requested.
            top_k: Requested top_k limit.

        Returns:
            Tuple of (parsed_results, usage).
            parsed_results: List of (index, score, optional_doc) tuples.
        """
        pass

    def make_request(self, url: str, payload: Json) -> Json:
        """
        Make HTTP request.

        Args:
            url: Request URL.
            payload: Request payload.

        Returns:
            Response JSON data.

        Raises:
            requests.RequestException: On network or HTTP errors.
        """
        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_s,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return response.json()

    async def amake_request(
        self, url: str, payload: Json, async_client: httpx.AsyncClient
    ) -> Json:
        """
        Make async HTTP request.

        Args:
            url: Request URL.
            payload: Request payload.
            async_client: httpx.AsyncClient instance.

        Returns:
            Response JSON data.

        Raises:
            httpx.HTTPError: On network or HTTP errors.
        """
        response = await async_client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_results(
        parsed_results: list[tuple[int, float, str | None]],
        include_docs: bool,
        top_k: int | None,
    ) -> Ranked | RankedWithDoc:
        """
        Normalize parsed results to unified format.

        Args:
            parsed_results: List of (index, score, optional_doc) tuples.
            include_docs: Whether to include documents.
            top_k: Limit results to top_k.

        Returns:
            Normalized results (Ranked or RankedWithDoc).
        """
        # Sort by score (descending - higher is better)
        parsed_results.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            parsed_results = parsed_results[:top_k]

        # Format results based on include_docs
        if include_docs:
            results: Ranked | RankedWithDoc = [
                (idx, score, doc)
                for idx, score, doc in parsed_results
                if doc is not None
            ]
        else:
            results = [(idx, score) for idx, score, _ in parsed_results]

        return results


class OpenAICompatibleHandler(RerankModeHandler):
    """
    Handler for OpenAI-compatible rerank API.

    Standard OpenAI rerank format:
    - Endpoint: POST {base_url}/rerank
    - Request: {"model": "...", "query": "...", "documents": [...], "top_n": ..., "return_documents": ...}
    - Response: {"results": [{"index": 0, "relevance_score": 0.95, "document": {"text": "..."}}, ...], "usage": {...}}
    """

    def build_request(
        self,
        query: str,
        docs: Sequence[str],
        model: str,
        top_k: int | None,
        include_docs: bool,
        extra: Json | None,
    ) -> tuple[str, Json]:
        """Build OpenAI-compatible request."""
        payload: Json = {
            "model": model,
            "query": query,
            "documents": list(docs),
        }

        if top_k is not None:
            payload["top_n"] = top_k

        if include_docs:
            payload["return_documents"] = True

        if extra:
            payload.update(extra)

        # Determine endpoint URL
        if "/rerank" in self.base_url:
            url = self.base_url
        else:
            url = f"{self.base_url}/rerank"

        return url, payload

    def parse_response(
        self,
        response_data: Json,
        docs: Sequence[str],
        include_docs: bool,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float, str | None]], Usage]:
        """Parse OpenAI-compatible response."""
        results_data = response_data.get("results", [])
        if not results_data:
            raise ValueError("No results in API response")

        parsed_results: list[tuple[int, float, str | None]] = []
        for item in results_data:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Unexpected result format: {item} (type: {type(item)})"
                )

            index = item.get("index", 0)
            score = item.get("relevance_score", 0.0)

            # Extract document text if available
            doc = None
            if include_docs:
                doc_obj = item.get("document")
                if isinstance(doc_obj, dict):
                    doc = doc_obj.get("text") or doc_obj.get("content")
                elif isinstance(doc_obj, str):
                    doc = doc_obj

            parsed_results.append((index, score, doc))

        usage = parse_usage(response_data)
        return parsed_results, usage


class DashScopeHandler(RerankModeHandler):
    """
    Handler for Alibaba Cloud DashScope rerank API.

    DashScope rerank format:
    - Endpoint: POST {base_url}/text-rerank/text-rerank (full path in base_url)
    - Request: {"model": "...", "input": {"query": "...", "documents": [...]}, "parameters": {...}}
    - Response: {"output": {"results": [...]}, "usage": {...}}
    """

    def build_request(
        self,
        query: str,
        docs: Sequence[str],
        model: str,
        top_k: int | None,
        include_docs: bool,
        extra: Json | None,
    ) -> tuple[str, Json]:
        """Build DashScope request."""
        payload: Json = {
            "model": model,
            "input": {
                "query": query,
                "documents": list(docs),
            },
        }

        # DashScope uses "parameters" for additional options
        if top_k is not None or include_docs or extra:
            parameters: Json = {}
            if top_k is not None:
                parameters["top_n"] = top_k
            if include_docs:
                parameters["return_documents"] = True
            if extra:
                parameters.update(extra)
            if parameters:
                payload["parameters"] = parameters

        # DashScope endpoint is typically the full path
        url = self.base_url

        return url, payload

    def parse_response(
        self,
        response_data: Json,
        docs: Sequence[str],
        include_docs: bool,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float, str | None]], Usage]:
        """Parse DashScope response."""
        output = response_data.get("output", {})
        results_data = output.get("results", [])
        if not results_data:
            raise ValueError("No results in API response")

        parsed_results: list[tuple[int, float, str | None]] = []
        for item in results_data:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Unexpected result format: {item} (type: {type(item)})"
                )

            index = item.get("index", 0)
            score = item.get("relevance_score", 0.0)

            # Extract document text if available
            doc = None
            if include_docs:
                doc_obj = item.get("document")
                if isinstance(doc_obj, dict):
                    doc = doc_obj.get("text") or doc_obj.get("content")
                elif isinstance(doc_obj, str):
                    doc = doc_obj

            parsed_results.append((index, score, doc))

        usage = parse_usage(response_data)
        return parsed_results, usage


class Rerank:
    """
    Rerank API client.

    Provides a simple, function-like API for document reranking.
    Supports two modes:
    - "openai": OpenAI-compatible standard API (default)
    - "dashscope": Alibaba Cloud DashScope API

    Examples:
        >>> # OpenAI-compatible mode (default)
        >>> rerank = Rerank(
        ...     base_url="https://api.example.com/v1",
        ...     api_key="key",
        ...     model="rerank-model",
        ...     mode="openai"
        ... )
        >>> result = rerank("python http", ["urllib", "requests", "httpx"])
        >>> ranked = result.results  # List[Tuple[int, float]]

        >>> # DashScope mode
        >>> rerank = Rerank(
        ...     base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        ...     api_key="key",
        ...     model="qwen3-rerank",
        ...     mode="dashscope"
        ... )
        >>> result = rerank("python http", ["urllib", "requests", "httpx"])
    """

    # Mode handler registry
    _HANDLERS: dict[str, type[RerankModeHandler]] = {
        "openai": OpenAICompatibleHandler,
        "dashscope": DashScopeHandler,
    }

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        mode: str = "openai",
        timeout_s: float = 60.0,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ):
        """
        Initialize Rerank client.

        Args:
            base_url: Base URL for the API (e.g., "https://api.example.com/v1").
            api_key: API key for authentication (optional if provided in headers).
            model: Default model to use (can be overridden in __call__).
            mode: Rerank mode. "openai" for OpenAI-compatible, "dashscope" for DashScope.
                  Default is "openai".
            timeout_s: Request timeout in seconds.
            headers: Additional headers to include in requests.
            proxies: Optional proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.

        Raises:
            ValueError: If mode is not supported.
        """
        if mode not in self._HANDLERS:
            available = ", ".join(f'"{m}"' for m in self._HANDLERS.keys())
            raise ValueError(f'Mode must be one of {available}, got "{mode}"')

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.timeout_s = timeout_s
        self.headers = headers or {}
        self.proxies = proxies  # None means use environment variables

        # Set default headers
        if self.api_key:
            self.headers.setdefault("Authorization", f"Bearer {self.api_key}")
        self.headers.setdefault("Content-Type", "application/json")

        # Initialize mode handler
        handler_class = self._HANDLERS[mode]
        self._handler = handler_class(
            base_url=self.base_url,
            api_key=self.api_key,
            headers=self.headers,
            timeout_s=self.timeout_s,
            proxies=self.proxies,
        )

        # Async client (lazy initialization)
        self._async_client: httpx.AsyncClient | None = None

    def __call__(
        self,
        query: str,
        docs: Sequence[str],
        *,
        model: str | None = None,
        top_k: int | None = None,
        include_docs: bool = False,
        extra: Json | None = None,
        return_raw: bool = False,
        mode: str | None = None,
    ) -> RerankResult:
        """
        Make a rerank request.

        Args:
            query: Query string.
            docs: Sequence of document strings to rerank.
            model: Model to use (overrides default).
            top_k: Number of top results to return (optional).
            include_docs: Whether to include documents in results.
            extra: Additional parameters to include in the request.
            return_raw: Whether to include full raw response.
            mode: Override mode for this call ("openai" or "dashscope").

        Returns:
            RerankResult with ranked results and usage.

        Raises:
            requests.RequestException: On network or HTTP errors.
            ValueError: On invalid input or response format.
        """
        if not docs:
            raise ValueError("Docs cannot be empty")

        # Prepare request
        model = model or self.model
        if not model:
            raise ValueError("Model must be specified (either in __init__ or __call__)")

        # Determine which mode to use
        use_mode = mode or self.mode
        if use_mode not in self._HANDLERS:
            available = ", ".join(f'"{m}"' for m in self._HANDLERS.keys())
            raise ValueError(f'Mode must be one of {available}, got "{use_mode}"')

        # Get or create handler for this call
        if use_mode == self.mode:
            handler = self._handler
        else:
            # Create temporary handler for mode override
            handler_class = self._HANDLERS[use_mode]
            handler = handler_class(
                base_url=self.base_url,
                api_key=self.api_key,
                headers=self.headers,
                timeout_s=self.timeout_s,
            )

        # Build request
        url, payload = handler.build_request(
            query=query,
            docs=docs,
            model=model,
            top_k=top_k,
            include_docs=include_docs,
            extra=extra,
        )

        # Make request
        response_data = handler.make_request(url, payload)

        # Parse response
        parsed_results, usage = handler.parse_response(
            response_data=response_data,
            docs=docs,
            include_docs=include_docs,
            top_k=top_k,
        )

        # Normalize results to unified format
        results = handler._normalize_results(parsed_results, include_docs, top_k)

        # Return result
        return RerankResult(
            results=results,
            usage=usage,
            raw=response_data if return_raw else {},
        )

    # =========================================================================
    # Async Methods
    # =========================================================================

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                headers=self.headers,
            )
        return self._async_client

    async def acall(
        self,
        query: str,
        docs: Sequence[str],
        *,
        model: str | None = None,
        top_k: int | None = None,
        include_docs: bool = False,
        extra: Json | None = None,
        return_raw: bool = False,
        mode: str | None = None,
    ) -> RerankResult:
        """
        Make an async rerank request.

        This is the async version of ``__call__()``. All parameters and behavior
        are identical to the sync version.

        Args:
            query: Query string.
            docs: Sequence of document strings to rerank.
            model: Model to use (overrides default).
            top_k: Number of top results to return (optional).
            include_docs: Whether to include documents in results.
            extra: Additional parameters to include in the request.
            return_raw: Whether to include full raw response.
            mode: Override mode for this call.

        Returns:
            RerankResult with ranked results and usage.

        Examples:
            >>> result = await rerank.acall("python http", ["urllib", "requests"])
            >>> ranked = result.results

            Concurrent rerank requests:
            >>> queries = ["python http", "python json", "python async"]
            >>> docs = ["requests", "json", "asyncio", "urllib"]
            >>> tasks = [rerank.acall(q, docs) for q in queries]
            >>> results = await asyncio.gather(*tasks)
        """
        if not docs:
            raise ValueError("Docs cannot be empty")

        model = model or self.model
        if not model:
            raise ValueError("Model must be specified (either in __init__ or acall)")

        use_mode = mode or self.mode
        if use_mode not in self._HANDLERS:
            available = ", ".join(f'"{m}"' for m in self._HANDLERS.keys())
            raise ValueError(f'Mode must be one of {available}, got "{use_mode}"')

        # Get or create handler
        if use_mode == self.mode:
            handler = self._handler
        else:
            handler_class = self._HANDLERS[use_mode]
            handler = handler_class(
                base_url=self.base_url,
                api_key=self.api_key,
                headers=self.headers,
                timeout_s=self.timeout_s,
            )

        # Build request
        url, payload = handler.build_request(
            query=query,
            docs=docs,
            model=model,
            top_k=top_k,
            include_docs=include_docs,
            extra=extra,
        )

        # Make async request
        client = self._get_async_client()
        response_data = await handler.amake_request(url, payload, client)

        # Parse response
        parsed_results, usage = handler.parse_response(
            response_data=response_data,
            docs=docs,
            include_docs=include_docs,
            top_k=top_k,
        )

        # Normalize results
        results = handler._normalize_results(parsed_results, include_docs, top_k)

        return RerankResult(
            results=results,
            usage=usage,
            raw=response_data if return_raw else {},
        )

    async def aclose(self) -> None:
        """Close the async client and release resources."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def close(self) -> None:
        """Close sync resources (placeholder for consistency)."""
        pass

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
