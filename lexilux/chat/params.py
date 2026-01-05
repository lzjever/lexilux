"""
Chat API parameter configuration classes.

Provides dataclass-based parameter configuration for OpenAI-compatible chat completions,
with support for standard parameters and custom extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ChatParams:
    """
    Standard parameters for chat completion requests.

    This class defines the most commonly used parameters for OpenAI-compatible
    chat completion APIs. All parameters are optional and have sensible defaults.

    Attributes:
        temperature: What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower
            values like 0.2 will make it more focused and deterministic.
            Default: 0.7

        top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability
            mass are considered.
            Range: 0.0 to 1.0. Default: 1.0

        max_tokens: The maximum number of tokens to generate in the chat completion.
            The total length of input tokens and generated tokens is limited by the
            model's context length.
            Default: None (no limit, up to model's maximum)

        stop: Up to 4 sequences where the API will stop generating further tokens.
            The returned text will not contain the stop sequence.
            Can be a single string or a list of strings.
            Default: None

        presence_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on whether they appear in the text so far, increasing
            the model's likelihood to talk about new topics.
            Default: 0.0

        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on their existing frequency in the text so far,
            decreasing the model's likelihood to repeat the same line verbatim.
            Default: 0.0

        logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a dictionary mapping token IDs (integers) to an
            associated bias value from -100 to 100. Values around -100 should
            decrease the likelihood of the token appearing, while values around 100
            should increase it.
            Default: None (empty dict)

        user: A unique identifier representing your end-user, which can help OpenAI
            to monitor and detect abuse. This is useful for tracking and rate limiting.
            Default: None

        n: How many chat completion choices to generate for each input message.
            Note: Most implementations return only the first choice. This parameter
            is included for compatibility but may not be fully supported by all
            providers.
            Default: 1

        extra: Additional custom parameters for OpenAI-compatible servers that may
            accept non-standard parameters. These will be merged into the request
            payload. Useful for provider-specific features.
            Default: None (empty dict)

    Examples:
        Basic usage with defaults:
        >>> params = ChatParams()
        >>> # temperature=0.7, top_p=1.0, etc.

        Custom temperature and max_tokens:
        >>> params = ChatParams(temperature=0.5, max_tokens=100)

        With stop sequences:
        >>> params = ChatParams(stop=["\\n\\n", "Human:"])

        With penalties:
        >>> params = ChatParams(
        ...     presence_penalty=0.6,
        ...     frequency_penalty=0.3
        ... )

        With custom parameters:
        >>> params = ChatParams(
        ...     temperature=0.8,
        ...     extra={"custom_param": "value", "another_param": 123}
        ... )
    """

    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: str | Sequence[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    user: str | None = None
    n: int = 1
    extra: dict[str, Any] | None = None

    def to_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """
        Convert parameters to dictionary for API request.

        Args:
            exclude_none: Whether to exclude None values from the output.
                Default: True

        Returns:
            Dictionary of parameters ready for API request.

        Examples:
            >>> params = ChatParams(temperature=0.5, max_tokens=100)
            >>> params.to_dict()
            {'temperature': 0.5, 'top_p': 1.0, 'max_tokens': 100, ...}
        """
        result: dict[str, Any] = {}

        # Add standard parameters
        if not exclude_none or self.temperature is not None:
            result["temperature"] = self.temperature
        if not exclude_none or self.top_p is not None:
            result["top_p"] = self.top_p
        if not exclude_none or self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if not exclude_none or self.stop is not None:
            if self.stop is not None:
                if isinstance(self.stop, str):
                    result["stop"] = [self.stop]
                else:
                    result["stop"] = list(self.stop)
        if not exclude_none or self.presence_penalty != 0.0:
            result["presence_penalty"] = self.presence_penalty
        if not exclude_none or self.frequency_penalty != 0.0:
            result["frequency_penalty"] = self.frequency_penalty
        if not exclude_none or self.logit_bias is not None:
            if self.logit_bias is not None:
                result["logit_bias"] = self.logit_bias
        if not exclude_none or self.user is not None:
            if self.user is not None:
                result["user"] = self.user
        if not exclude_none or self.n != 1:
            result["n"] = self.n

        # Merge extra parameters
        if self.extra:
            result.update(self.extra)

        return result
