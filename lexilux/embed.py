"""
Embedding API client.

Provides a simple, function-like API for text embeddings with unified usage tracking.
Supports both sync and async operations with connection pooling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import requests

from lexilux._async_client import AsyncClientMixin
from lexilux.embed_params import EmbedParams
from lexilux.usage import Json, ResultBase, Usage
from lexilux.chat.utils import parse_usage

if TYPE_CHECKING:
    pass

# Type alias
Vector = list[float]


class EmbedResult(ResultBase):
    """
    Embedding result.

    The vectors field contains:
    - Single Vector (List[float]) when input is a single string
    - List[Vector] (List[List[float]]) when input is a sequence of strings

    Attributes:
        vectors: Embedding vector(s).
        usage: Usage statistics.
        raw: Raw API response.

    Examples:
        >>> result = embed("Hello")
        >>> vector = result.vectors  # List[float]

        >>> result = embed(["Hello", "World"])
        >>> vectors = result.vectors  # List[List[float]]
    """

    def __init__(
        self,
        *,
        vectors: Vector | list[Vector],
        usage: Usage,
        raw: Json | None = None,
    ):
        """
        Initialize EmbedResult.

        Args:
            vectors: Embedding vector(s).
            usage: Usage statistics.
            raw: Raw API response.
        """
        super().__init__(usage=usage, raw=raw)
        self.vectors = vectors

    def __repr__(self) -> str:
        """Return string representation."""
        if isinstance(self.vectors[0], list):
            # List of vectors
            return f"EmbedResult(vectors=[{len(self.vectors)} vectors], usage={self.usage!r})"
        else:
            # Single vector
            return (
                f"EmbedResult(vectors=[{len(self.vectors)} dims], usage={self.usage!r})"
            )


class Embed(AsyncClientMixin):
    """
    Embedding API client with connection pooling.

    Provides a simple, function-like API for text embeddings.
    Uses connection pooling for improved performance in high-throughput scenarios.

    Examples:
        >>> embed = Embed(base_url="https://api.example.com/v1", api_key="key", model="text-embedding-ada-002")
        >>> result = embed("Hello, world!")
        >>> vector = result.vectors  # List[float]

        >>> result = embed(["text1", "text2"])
        >>> vectors = result.vectors  # List[List[float]]

        >>> # Context manager for proper resource cleanup
        >>> with Embed(base_url="...", api_key="key") as embed:
        ...     result = embed("Hello")
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
        pool_size: int = 10,
    ):
        """
        Initialize Embed client.

        Args:
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1").
            api_key: API key for authentication (optional if provided in headers).
            model: Default model to use (can be overridden in __call__).
            timeout_s: Request timeout in seconds.
            headers: Additional headers to include in requests.
            proxies: Optional proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.
            pool_size: Connection pool size for HTTP adapter (default: 10).
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.headers = headers or {}
        self.proxies = proxies  # None means use environment variables

        # Set default headers
        if self.api_key:
            self.headers.setdefault("Authorization", f"Bearer {self.api_key}")
        self.headers.setdefault("Content-Type", "application/json")

        # Create Session with connection pooling for sync requests
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Initialize async client (lazy) - required by AsyncClientMixin
        self._async_client = None

    def _build_payload(
        self,
        input: str | Sequence[str],
        model: str | None,
        dimensions: int | None,
        encoding_format: Literal["float", "base64"] | None,
        user: str | None,
        params: EmbedParams | None,
        extra: Json | None,
    ) -> tuple[Json, bool]:
        """Build the payload for the embeddings request and identify input type."""
        is_single = isinstance(input, str)
        input_list = [input] if is_single else list(input)
        if not input_list:
            raise ValueError("Input cannot be empty")

        final_model = model or self.model
        if not final_model:
            raise ValueError("Model must be specified (either in __init__ or in call)")

        # Start with params object if provided
        param_dict = params.to_dict(exclude_none=True) if params else {}

        # Individual arguments override what's in params
        if dimensions is not None:
            param_dict["dimensions"] = dimensions
        if encoding_format is not None:
            param_dict["encoding_format"] = encoding_format
        if user is not None:
            param_dict["user"] = user

        # Build final payload
        payload: Json = {"model": final_model, "input": input_list, **param_dict}

        # Merge extra parameters, which have the highest precedence
        if extra:
            payload.update(extra)

        return payload, is_single

    def _process_response(
        self, response_data: Json, is_single: bool, return_raw: bool
    ) -> EmbedResult:
        """Process the raw API response into an EmbedResult."""
        data_list = response_data.get("data", [])
        if not data_list:
            raise ValueError("No data in API response")

        vectors: list[Vector] = [item["embedding"] for item in data_list]

        result_vectors: Vector | list[Vector] = vectors[0] if is_single else vectors
        usage = parse_usage(response_data)

        return EmbedResult(
            vectors=result_vectors,
            usage=usage,
            raw=response_data if return_raw else {},
        )

    def __call__(
        self,
        input: str | Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        user: str | None = None,
        params: EmbedParams | None = None,
        extra: Json | None = None,
        return_raw: bool = False,
    ) -> EmbedResult:
        """
        Make an embedding request.

        Supports both direct parameter passing (backward compatible) and EmbedParams
        dataclass for structured configuration.

        Args:
            input: Single text string or sequence of text strings.
            model: Model to use (overrides default).
            dimensions: Number of dimensions for output embeddings. Only supported
                in some models (e.g., ``text-embedding-3-*``). Default: None (use model default)
            encoding_format: Format to return embeddings. "float" (default) or "base64".
                Some providers may support additional formats.
            user: Unique identifier for end-user (for monitoring/rate limiting).
            params: EmbedParams dataclass instance. If provided, overrides individual
                parameters above. Useful for structured configuration.
            extra: Additional custom parameters for non-standard providers.
                Merged with params if both are provided.
            return_raw: Whether to include full raw response.

        Returns:
            EmbedResult with vectors and usage.

        Raises:
            requests.RequestException: On network or HTTP errors.
            ValueError: On invalid input or response format.

        Examples:
            Basic usage (backward compatible):
            >>> result = embed("Hello", dimensions=512)

            Using EmbedParams:
            >>> from lexilux import EmbedParams
            >>> params = EmbedParams(dimensions=512, encoding_format="float")
            >>> result = embed("Hello", params=params)

            Combining params and extra:
            >>> result = embed("Hello", params=params, extra={"custom": "value"})
        """
        payload, is_single = self._build_payload(
            input, model, dimensions, encoding_format, user, params, extra
        )

        url = f"{self.base_url}/embeddings"
        # Use session with connection pooling
        response = self._session.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_s,
            proxies=self.proxies,
        )
        response.raise_for_status()

        response_data = response.json()
        return self._process_response(response_data, is_single, return_raw)

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def acall(
        self,
        input: str | Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        user: str | None = None,
        params: EmbedParams | None = None,
        extra: Json | None = None,
        return_raw: bool = False,
    ) -> EmbedResult:
        """
        Make an async embedding request.

        This is the async version of ``__call__()``. All parameters and behavior
        are identical to the sync version.

        Args:
            input: Single text string or sequence of text strings.
            model: Model to use (overrides default).
            dimensions: Number of dimensions for output embeddings.
            encoding_format: Format to return embeddings.
            user: Unique identifier for end-user.
            params: EmbedParams dataclass instance.
            extra: Additional custom parameters.
            return_raw: Whether to include full raw response.

        Returns:
            EmbedResult with vectors and usage.

        Examples:
            >>> result = await embed.acall("Hello")
            >>> vector = result.vectors
        """
        payload, is_single = self._build_payload(
            input, model, dimensions, encoding_format, user, params, extra
        )

        url = f"{self.base_url}/embeddings"
        client = self._get_async_client()
        response = await client.post(url, json=payload)
        response.raise_for_status()

        response_data = response.json()
        return self._process_response(response_data, is_single, return_raw)

    def close(self) -> None:
        """
        Close the sync session and release resources.

        Should be called when done with the client, or use context manager.
        """
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()
            self._session = None
