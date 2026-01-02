"""
Rerank API client.

Provides a simple, function-like API for document reranking with unified usage tracking.
Supports multiple provider modes: OpenAI-compatible and DashScope.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import requests

from lexilux.usage import Usage, ResultBase, Json

if TYPE_CHECKING:
    pass

# Type aliases
Ranked = List[Tuple[int, float]]  # (index, score)
RankedWithDoc = List[Tuple[int, float, str]]  # (index, score, doc)


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
        results: Union[Ranked, RankedWithDoc],
        usage: Usage,
        raw: Optional[Json] = None,
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
        return f"RerankResult(results=[{len(self.results)} items], usage={self.usage!r})"


class RerankModeHandler(ABC):
    """
    Abstract base class for rerank mode handlers.

    Each handler implements provider-specific request/response format conversion
    while maintaining a unified interface.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        headers: Dict[str, str],
        timeout_s: float,
    ):
        """
        Initialize mode handler.

        Args:
            base_url: Base URL for the API.
            api_key: API key for authentication.
            headers: HTTP headers.
            timeout_s: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = headers
        self.timeout_s = timeout_s

    @abstractmethod
    def build_request(
        self,
        query: str,
        docs: Sequence[str],
        model: str,
        top_k: Optional[int],
        include_docs: bool,
        extra: Optional[Json],
    ) -> Tuple[str, Json]:
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
        top_k: Optional[int],
    ) -> Tuple[List[Tuple[int, float, Optional[str]]], Usage]:
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
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_usage(response_data: Json) -> Usage:
        """
        Parse usage information from API response.

        Args:
            response_data: API response data.

        Returns:
            Usage object.
        """
        usage_data = response_data.get("usage", {})
        return Usage(
            input_tokens=usage_data.get("prompt_tokens") or usage_data.get("input_tokens"),
            output_tokens=usage_data.get("completion_tokens") or usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            details=usage_data,
        )

    @staticmethod
    def _normalize_results(
        parsed_results: List[Tuple[int, float, Optional[str]]],
        include_docs: bool,
        top_k: Optional[int],
    ) -> Union[Ranked, RankedWithDoc]:
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
            results: Union[Ranked, RankedWithDoc] = [
                (idx, score, doc) for idx, score, doc in parsed_results if doc is not None
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
        top_k: Optional[int],
        include_docs: bool,
        extra: Optional[Json],
    ) -> Tuple[str, Json]:
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
        top_k: Optional[int],
    ) -> Tuple[List[Tuple[int, float, Optional[str]]], Usage]:
        """Parse OpenAI-compatible response."""
        results_data = response_data.get("results", [])
        if not results_data:
            raise ValueError("No results in API response")

        parsed_results: List[Tuple[int, float, Optional[str]]] = []
        for item in results_data:
            if not isinstance(item, dict):
                raise ValueError(f"Unexpected result format: {item} (type: {type(item)})")

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

        usage = self._parse_usage(response_data)
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
        top_k: Optional[int],
        include_docs: bool,
        extra: Optional[Json],
    ) -> Tuple[str, Json]:
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
        top_k: Optional[int],
    ) -> Tuple[List[Tuple[int, float, Optional[str]]], Usage]:
        """Parse DashScope response."""
        output = response_data.get("output", {})
        results_data = output.get("results", [])
        if not results_data:
            raise ValueError("No results in API response")

        parsed_results: List[Tuple[int, float, Optional[str]]] = []
        for item in results_data:
            if not isinstance(item, dict):
                raise ValueError(f"Unexpected result format: {item} (type: {type(item)})")

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

        usage = self._parse_usage(response_data)
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
    _HANDLERS: Dict[str, type[RerankModeHandler]] = {
        "openai": OpenAICompatibleHandler,
        "dashscope": DashScopeHandler,
    }

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mode: str = "openai",
        timeout_s: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
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
        )

    def __call__(
        self,
        query: str,
        docs: Sequence[str],
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        include_docs: bool = False,
        extra: Optional[Json] = None,
        return_raw: bool = False,
        mode: Optional[str] = None,
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
