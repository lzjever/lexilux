"""
Unified exception hierarchy for Lexilux.

All exceptions inherit from LexiluxError and include error codes for
programmatic error handling.
"""

from __future__ import annotations


class LexiluxError(Exception):
    """
    Base exception class for all Lexilux errors.

    All Lexilux exceptions inherit from this class, allowing users to
    catch all Lexilux-specific errors with a single except clause.

    Attributes:
        message: Human-readable error message.
        code: Machine-readable error code (e.g., "authentication_failed").
        retryable: Whether the error is retryable (transient) or permanent.

    Examples:
        >>> try:
        ...     result = chat("Hello")
        ... except LexiluxError as e:
        ...     if e.retryable:
        ...         print(f"Temporary error {e.code}: {e.message}")
        ...     else:
        ...     print(f"Permanent error {e.code}: {e.message}")
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        retryable: bool = False,
    ):
        """
        Initialize LexiluxError.

        Args:
            message: Human-readable error message.
            code: Machine-readable error code (defaults to class name).
            retryable: Whether this is a transient error (default: False).
        """
        self.message = message
        self.code = code or self.__class__.__name__
        self.retryable = retryable
        super().__init__(self.message)

    def __repr__(self) -> str:
        """Return representation including code."""
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class APIError(LexiluxError):
    """
    Base class for API-related errors.

    Raised when the API returns an error response.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        code: str | None = None,
        retryable: bool = False,
    ):
        """
        Initialize APIError.

        Args:
            message: Error message from API or generated.
            status_code: HTTP status code (if applicable).
            code: Error code.
            retryable: Whether the error is retryable.
        """
        super().__init__(message, code, retryable)
        self.status_code = status_code


class AuthenticationError(APIError):
    """
    Authentication or authorization failed.

    Raised when the API key is invalid, expired, or lacks permissions.

    This error is NOT retryable without fixing the credentials.
    """

    code = "authentication_failed"
    retryable = False

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401, code=self.code, retryable=self.retryable)


class RateLimitError(APIError):
    """
    Rate limit exceeded.

    Raised when too many requests are sent in a short period.

    This error IS retryable after a delay.
    """

    code = "rate_limit_exceeded"
    retryable = True

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, code=self.code, retryable=self.retryable)


class InvalidRequestError(APIError):
    """
    Invalid request to the API.

    Raised when the request is malformed or contains invalid parameters.

    This error is NOT retryable without fixing the request.
    """

    code = "invalid_request"
    retryable = False

    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, status_code=400, code=self.code, retryable=self.retryable)


class NotFoundError(APIError):
    """
    Resource not found.

    Raised when a requested resource (model, endpoint, etc.) doesn't exist.

    This error is NOT retryable.
    """

    code = "not_found"
    retryable = False

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404, code=self.code, retryable=self.retryable)


class ServerError(APIError):
    """
    Internal server error from the API.

    Raised when the API server encounters an unexpected error.

    This error IS retryable.
    """

    code = "server_error"
    retryable = True

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, status_code=500, code=self.code, retryable=self.retryable)


class NetworkError(LexiluxError):
    """
    Network-related error.

    Base class for connection, timeout, and DNS errors.
    """

    retryable = True


class TimeoutError(NetworkError):
    """
    Request timeout.

    Raised when the request takes too long to complete.

    This error IS retryable.
    """

    code = "timeout"
    retryable = True

    def __init__(self, message: str = "Request timeout"):
        super().__init__(message, code=self.code, retryable=self.retryable)


class ConnectionError(NetworkError):
    """
    Connection failed.

    Raised when unable to establish a connection to the server.

    This error IS retryable.
    """

    code = "connection_failed"
    retryable = True

    def __init__(self, message: str = "Connection failed"):
        super().__init__(message, code=self.code, retryable=self.retryable)


class ValidationError(LexiluxError):
    """
    Input validation error.

    Raised when client-side validation fails (e.g., invalid parameters).

    This error is NOT retryable without fixing the input.
    """

    code = "validation_error"
    retryable = False

    def __init__(self, message: str = "Validation failed"):
        super().__init__(message, code=self.code, retryable=self.retryable)


class ConfigurationError(LexiluxError):
    """
    Configuration error.

    Raised when the client is misconfigured (e.g., missing required parameters).

    This error is NOT retryable without fixing the configuration.
    """

    code = "configuration_error"
    retryable = False

    def __init__(self, message: str = "Configuration error"):
        super().__init__(message, code=self.code, retryable=self.retryable)
