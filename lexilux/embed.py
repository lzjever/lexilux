"""
Embedding API client.

Provides a simple, function-like API for text embeddings with unified usage tracking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import requests

from lexilux.usage import Usage, ResultBase, Json

if TYPE_CHECKING:
    pass

# Type alias
Vector = List[float]


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
        vectors: Union[Vector, List[Vector]],
        usage: Usage,
        raw: Optional[Json] = None,
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
            return f"EmbedResult(vectors=[{len(self.vectors)} dims], usage={self.usage!r})"


class Embed:
    """
    Embedding API client.

    Provides a simple, function-like API for text embeddings.

    Examples:
        >>> embed = Embed(base_url="https://api.example.com/v1", api_key="key", model="text-embedding-ada-002")
        >>> result = embed("Hello, world!")
        >>> vector = result.vectors  # List[float]

        >>> result = embed(["text1", "text2"])
        >>> vectors = result.vectors  # List[List[float]]
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize Embed client.

        Args:
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1").
            api_key: API key for authentication (optional if provided in headers).
            model: Default model to use (can be overridden in __call__).
            timeout_s: Request timeout in seconds.
            headers: Additional headers to include in requests.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.headers = headers or {}

        # Set default headers
        if self.api_key:
            self.headers.setdefault("Authorization", f"Bearer {self.api_key}")
        self.headers.setdefault("Content-Type", "application/json")

    def _parse_usage(self, response_data: Json) -> Usage:
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

    def __call__(
        self,
        input: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        extra: Optional[Json] = None,
        return_raw: bool = False,
    ) -> EmbedResult:
        """
        Make an embedding request.

        Args:
            input: Single text string or sequence of text strings.
            model: Model to use (overrides default).
            extra: Additional parameters to include in the request.
            return_raw: Whether to include full raw response.

        Returns:
            EmbedResult with vectors and usage.

        Raises:
            requests.RequestException: On network or HTTP errors.
            ValueError: On invalid input or response format.
        """
        # Normalize input to list
        is_single = isinstance(input, str)
        input_list = [input] if is_single else list(input)

        if not input_list:
            raise ValueError("Input cannot be empty")

        # Prepare request
        model = model or self.model
        if not model:
            raise ValueError("Model must be specified (either in __init__ or __call__)")

        payload: Json = {
            "model": model,
            "input": input_list,
        }

        if extra:
            payload.update(extra)

        # Make request
        url = f"{self.base_url}/embeddings"
        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_s,
        )
        response.raise_for_status()

        response_data = response.json()

        # Parse response
        data_list = response_data.get("data", [])
        if not data_list:
            raise ValueError("No data in API response")

        # Extract vectors
        vectors: List[Vector] = [item["embedding"] for item in data_list]

        # Return single vector or list of vectors based on input
        result_vectors: Union[Vector, List[Vector]] = vectors[0] if is_single else vectors

        # Parse usage
        usage = self._parse_usage(response_data)

        # Return result
        return EmbedResult(
            vectors=result_vectors,
            usage=usage,
            raw=response_data if return_raw else {},
        )
