"""
Client-side rate limiting support.

Provides a rate limiter wrapper to prevent API quota exhaustion by limiting
the rate of outgoing requests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    from aiolimiter import AsyncLimiter

    AIOLIMITER_AVAILABLE = True
except ImportError:
    AIOLIMITER_AVAILABLE = False
    AsyncLimiter = None  # type: ignore


class RateLimiter:
    """
    Rate limiter for API requests.

    Wraps aiolimiter.AsyncLimiter to provide client-side rate limiting
    and prevent API quota exhaustion. Falls back gracefully when aiolimiter
    is not installed.

    Args:
        max_rate: Maximum number of requests allowed in the time period.
        time_period: Time period in seconds for the rate limit.

    Examples:
        >>> limiter = RateLimiter(max_rate=10, time_period=60)
        >>> await limiter.acquire()  # Wait if rate limit reached
        >>> # Make API request...
    """

    def __init__(self, *, max_rate: int, time_period: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            max_rate: Maximum number of requests allowed in the time period.
            time_period: Time period in seconds for the rate limit.

        Raises:
            ImportError: If aiolimiter is not installed and rate limiting is requested.
        """
        if not AIOLIMITER_AVAILABLE:
            raise ImportError(
                "aiolimiter is required for rate limiting. "
                "Install it with: pip install 'lexilux[rate-limit]'"
            )

        self._max_rate = max_rate
        self._time_period = time_period
        self._limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)

        logger.debug(
            "Rate limiter initialized: max_rate=%d, time_period=%s",
            max_rate,
            time_period,
        )

    @property
    def max_rate(self) -> int:
        """Get the maximum rate."""
        return self._max_rate

    @property
    def time_period(self) -> float:
        """Get the time period."""
        return self._time_period

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Blocks if the rate limit has been reached until capacity is available.

        Examples:
            >>> await limiter.acquire()
            >>> # Now safe to make API request
        """
        await self._limiter.acquire()
        logger.debug("Rate limiter: acquired permission for request")

    def is_available(self) -> bool:
        """
        Check if the rate limiter is available.

        Returns:
            True if aiolimiter is installed and rate limiting is available.
        """
        return AIOLIMITER_AVAILABLE


def has_rate_limit_support() -> bool:
    """
    Check if rate limiting support is available.

    Returns:
        True if aiolimiter is installed.
    """
    return AIOLIMITER_AVAILABLE
