"""
Usage and ResultBase classes for unified API responses.

All API calls return ResultBase subclasses that include usage statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Type alias for JSON-like data
Json = dict[str, Any]


class Usage:
    """
    Unified usage statistics (aligned with OpenAI completion/chat.completions usage semantics).

    All fields are optional to handle cases where the server doesn't provide usage information.
    The details field allows for extensibility (e.g., cached_tokens, reasoning_tokens).

    Attributes:
        input_tokens: Number of input tokens (optional).
        output_tokens: Number of output tokens (optional).
        total_tokens: Total number of tokens (optional).
        details: Additional usage details as a dictionary (optional).

    Examples:
        >>> usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        >>> print(usage.total_tokens)
        30

        >>> usage = Usage()  # Empty usage (server didn't provide)
        >>> print(usage.total_tokens)
        None
    """

    def __init__(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        details: Json | None = None,
    ):
        """
        Initialize Usage object.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            total_tokens: Total number of tokens.
            details: Additional usage details dictionary.
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.details = details or {}

    def __repr__(self) -> str:
        """Return string representation of Usage."""
        parts = []
        if self.input_tokens is not None:
            parts.append(f"input_tokens={self.input_tokens}")
        if self.output_tokens is not None:
            parts.append(f"output_tokens={self.output_tokens}")
        if self.total_tokens is not None:
            parts.append(f"total_tokens={self.total_tokens}")
        if self.details:
            parts.append(f"details={self.details}")
        return f"Usage({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Usage object."""
        if not isinstance(other, Usage):
            return False
        return (
            self.input_tokens == other.input_tokens
            and self.output_tokens == other.output_tokens
            and self.total_tokens == other.total_tokens
            and self.details == other.details
        )


class ResultBase:
    """
    Base class for all API results.

    All API calls return ResultBase subclasses, ensuring that result.usage is always available.
    The raw field stores the original API response for advanced use cases.

    Attributes:
        usage: Usage statistics object.
        raw: Raw API response as a dictionary (optional).

    Examples:
        >>> result = ChatResult(text="Hello", usage=Usage(total_tokens=10))
        >>> print(result.usage.total_tokens)
        10
        >>> print(result.raw)  # Access raw response if needed
        {}
    """

    def __init__(self, *, usage: Usage, raw: Json | None = None):
        """
        Initialize ResultBase object.

        Args:
            usage: Usage statistics object.
            raw: Raw API response dictionary.
        """
        self.usage = usage
        self.raw = raw or {}

    def __repr__(self) -> str:
        """Return string representation of ResultBase."""
        return f"{self.__class__.__name__}(usage={self.usage!r}, raw={self.raw!r})"
