"""
Chat API client.

Provides a simple, function-like API for chat completions with support for
both non-streaming and streaming responses.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from lexilux._base import BaseAPIClient
from lexilux._rate_limit import RateLimiter
from lexilux.chat._request import (
    SSEChatStreamParser,
    build_api_messages,
    build_params_dict,
    build_payload,
    parse_chat_completion_response,
    prepare_messages_for_request,
)
from lexilux.chat.continuer import ConversationContinuer
from lexilux.chat.reasoning import (
    build_reasoning_request,
    normalize_reasoning,
)
from lexilux.providers.registry import detect_provider_from_url

from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike
from lexilux.chat.params import ChatParams
from lexilux.chat.streaming import (
    AsyncStreamingIterator,
    StreamingIterator,
)
from lexilux.chat.validation import (
    validate_chat_params,
    validate_messages,
    validate_model,
    validate_stop,
)
from lexilux.usage import Json

if TYPE_CHECKING:
    from lexilux.chat.tools import Tool

logger = logging.getLogger(__name__)


def _get_original_prompt(messages: MessagesLike) -> str:
    return messages if isinstance(messages, str) else str(messages)


class Chat(BaseAPIClient):
    """
    Chat API client.

    Provides a simple, function-like API for chat completions with support for
    both non-streaming and streaming responses.

    **Important:** Chat is STATELESS - each call is independent. For multi-turn
    conversations, use ChatHistory to manage context and pass it via the
    `history` parameter.

    Method Overview:
        - `chat()` / `acall()`: Single request (may be truncated)
        - `stream()` / `astream()`: Streaming response (may be truncated)
        - `complete()` / `acomplete()`: Auto-continue if truncated
        - `complete_stream()` / `acomplete_stream()`: Streaming + auto-continue

    Related Classes:
        - ChatHistory: Manages conversation state (pass via `history` parameter)
        - Conversation: Low-level utility for handling truncated responses
            (use `chat.complete()` instead for simplicity)

    Examples:
        >>> # Simple single-turn query
        >>> chat = Chat(base_url="...", api_key="...", model="gpt-4")
        >>> result = chat("Hello, world!")
        >>> print(result.text)

        >>> # Streaming
        >>> for chunk in chat.stream("Tell me a joke"):
        ...     print(chunk.delta, end="")

        >>> # Multi-turn conversation (use ChatHistory)
        >>> from lexilux import ChatHistory
        >>> history = ChatHistory(system="You are helpful")
        >>> history.add_user("My name is Alice")
        >>> result = chat(history.get_messages())
        >>> history.add_assistant(result.text)
        >>> history.add_user("What's my name?")
        >>> result = chat(history.get_messages())  # AI remembers!

        >>> # Long content (auto-continue)
        >>> result = chat.complete("Write an essay", max_tokens=100)
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        max_retries: int = 0,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
        rate_limit: tuple[int, float] | None = None,
    ):
        """
        Initialize Chat client.

        Args:
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1").
            api_key: API key for authentication (optional if provided in headers).
            model: Default model to use (can be overridden in __call__).
            timeout_s: Request timeout in seconds (default for both connect and read).
            connect_timeout_s: Connection timeout in seconds (overrides timeout_s).
            read_timeout_s: Read timeout in seconds (overrides timeout_s).
            max_retries: Maximum number of retries for failed requests (default: 0).
            headers: Additional headers to include in requests.
            proxies: Optional proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.
            rate_limit: Optional rate limiting as (max_rate, time_period) tuple.
                    Example: (10, 60.0) for 10 requests per 60 seconds.
                    Requires aiolimiter to be installed.

        Note:
            Each HTTP request creates a new connection that closes after completion.
        """
        # Initialize base client
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            max_retries=max_retries,
            headers=headers,
            proxies=proxies,
        )

        # Chat-specific attributes
        self.model = model

        # Rate limiting
        self._rate_limiter: RateLimiter | None = None
        if rate_limit is not None:
            max_rate, time_period = rate_limit
            self._rate_limiter = RateLimiter(max_rate=max_rate, time_period=time_period)

        # Conversation continuer for complete() methods
        self._continuer = ConversationContinuer(self)

    @property
    def timeout_s(self) -> float:
        """
        Backward compatibility property for timeout.

        Returns the timeout value (or read timeout if tuple).
        """
        if isinstance(self.timeout, tuple):
            return self.timeout[1]  # Return read timeout
        return self.timeout

    def _build_payload(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None,
        model: str | None,
        system: str | None,
        params: ChatParams | None,
        extra: Json | None,
        stream: bool,
        include_usage: bool,
        reasoning: bool | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Json:
        """
        Build request payload (read-only from history, no cloning).

        This is the fast path for basic __call__ and stream operations.
        History is only read, never modified.
        """
        # Validate model
        final_model = validate_model(model, self.model)

        # Build messages (read-only from history)
        api_messages = build_api_messages(messages, system=system, history=history)
        validate_messages(api_messages)

        # Validate chat parameters
        validate_chat_params(
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            presence_penalty=kwargs.get("presence_penalty"),
            frequency_penalty=kwargs.get("frequency_penalty"),
        )

        # Validate stop sequences
        stop = kwargs.get("stop")
        if stop is not None:
            validated_stop = validate_stop(stop)
            kwargs["stop"] = validated_stop

        param_dict = build_params_dict(params=params, **kwargs)

        # Build reasoning params if enabled
        reasoning_params = {}
        if reasoning is not None:
            provider_id = detect_provider_from_url(self.base_url)
            if provider_id:
                normalized = normalize_reasoning(reasoning)
                reasoning_params = build_reasoning_request(provider_id, normalized)

        # Merge reasoning params into param_dict
        if reasoning_params:
            param_dict.update(reasoning_params)

        return build_payload(
            model=final_model,
            messages=api_messages,
            params=param_dict,
            stream=stream,
            include_usage=include_usage,
            extra=extra,
        )

    def _prepare_chat_request_with_history(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None,
        model: str | None,
        system: str | None,
        params: ChatParams | None,
        extra: Json | None,
        stream: bool,
        include_usage: bool,
        clone_history: bool = True,
        **kwargs: Any,
    ) -> tuple[Json, ChatHistory | None]:
        """
        Prepare request payload with mutable history tracking.

        This is for complete() family methods that need to track
        conversation state across multiple API calls.
        """
        # Validate model
        final_model = validate_model(model, self.model)

        normalized_messages, working_history, user_messages_to_add = (
            prepare_messages_for_request(
                messages,
                system=system,
                history=history,
                clone_history=clone_history,
            )
        )

        # Validate messages
        validate_messages(normalized_messages)

        # Validate chat parameters
        validate_chat_params(
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            presence_penalty=kwargs.get("presence_penalty"),
            frequency_penalty=kwargs.get("frequency_penalty"),
        )

        # Validate stop sequences
        stop = kwargs.get("stop")
        if stop is not None:
            validated_stop = validate_stop(stop)
            kwargs["stop"] = validated_stop

        param_dict = build_params_dict(params=params, **kwargs)
        payload = build_payload(
            model=final_model,
            messages=normalized_messages,
            params=param_dict,
            stream=stream,
            include_usage=include_usage,
            extra=extra,
        )

        if working_history:
            for user_msg in user_messages_to_add:
                working_history.add_user(user_msg)

        return payload, working_history

    def _process_chat_response_with_history(
        self,
        response_data: Json,
        working_history: ChatHistory | None,
        return_raw: bool,
    ) -> ChatResult:
        """Process response and update working history (for complete() family)."""
        result = parse_chat_completion_response(response_data, return_raw=return_raw)
        if working_history:
            working_history.append_result(result)
        return result

    def __call__(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        user: str | None = None,
        n: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | Any | None = None,
        parallel_tool_calls: bool | None = None,
        params: ChatParams | None = None,
        extra: Json | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        return_raw: bool = False,
    ) -> ChatResult:
        """
        Make a single chat completion request.

        History is read-only - used for context but never modified.
        """
        payload = self._build_payload(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=False,
            include_usage=False,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            n=n,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        response = self._make_request("chat/completions", payload)
        return parse_chat_completion_response(response.json(), return_raw=return_raw)

    def stream(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        user: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | Any | None = None,
        parallel_tool_calls: bool | None = None,
        params: ChatParams | None = None,
        extra: Json | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        include_usage: bool = True,
        return_raw_events: bool = False,
        include_reasoning: bool = False,
    ) -> StreamingIterator:
        """
        Stream a single chat completion response.

        History is read-only - used for context but never modified.
        """
        payload = self._build_payload(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=True,
            include_usage=include_usage,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            n=None,  # n>1 is not supported in streaming
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Create internal chunk generator using context manager
        def _chunk_generator() -> Iterator[ChatStreamChunk]:
            """Internal generator for streaming chunks."""
            parser = SSEChatStreamParser(
                return_raw_events=return_raw_events,
                include_reasoning=include_reasoning,
            )
            # Use context manager to ensure response is always closed
            with self._streaming_request_context(
                "chat/completions", payload
            ) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        line_str = line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    chunk = parser.feed_line(line_str)
                    if chunk is None:
                        continue
                    yield chunk
                    if parser.done:
                        break

        # Create and return iterator (no cleanup wrapper needed)
        return StreamingIterator(_chunk_generator())

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def _amake_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> Any:
        """
        Send async POST request to API endpoint with rate limiting.

        Overrides base method to add rate limiting if configured.

        Args:
            endpoint: API endpoint (e.g., "chat/completions").
            payload: Request body as dict.

        Returns:
            httpx.Response object.

        Raises:
            LexiluxTimeoutError: On timeout.
            LexiluxConnectionError: On connection failure.
            AuthenticationError: On authentication failure.
            RateLimitError: On rate limit exceeded.
            APIError: On other API errors.
            ValidationError: On invalid input.
        """
        # Apply rate limiting if configured
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        # Call parent's _amake_request
        return await super()._amake_request(endpoint, payload)

    async def acall(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        user: str | None = None,
        n: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | Any | None = None,
        parallel_tool_calls: bool | None = None,
        params: ChatParams | None = None,
        extra: Json | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        return_raw: bool = False,
    ) -> ChatResult:
        """
        Make an async chat completion request.

        History is read-only - used for context but never modified.
        """
        payload = self._build_payload(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=False,
            include_usage=False,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            n=n,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Make async request
        response = await self._amake_request("chat/completions", payload)
        return parse_chat_completion_response(response.json(), return_raw=return_raw)

    async def astream(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        user: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | Any | None = None,
        parallel_tool_calls: bool | None = None,
        params: ChatParams | None = None,
        extra: Json | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        include_usage: bool = True,
        return_raw_events: bool = False,
        include_reasoning: bool = False,
    ) -> AsyncStreamingIterator:
        """
        Stream an async chat completion response.

        History is read-only - used for context but never modified.
        """
        payload = self._build_payload(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=True,
            include_usage=include_usage,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            n=None,  # n>1 is not supported in streaming
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        async def _async_chunk_generator() -> AsyncIterator[ChatStreamChunk]:
            # Apply rate limiting if configured (before starting stream)
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()

            parser = SSEChatStreamParser(
                return_raw_events=return_raw_events,
                include_reasoning=include_reasoning,
            )
            stream = self._amake_streaming_request("chat/completions", payload)
            try:
                async for line in stream:
                    chunk = parser.feed_line(line)
                    if chunk is None:
                        continue
                    yield chunk
                    if parser.done:
                        break
            finally:
                logger.debug(
                    "Closing async streaming response and releasing connection"
                )
                await stream.aclose()

        # Create and return async iterator (no cleanup wrapper needed)
        return AsyncStreamingIterator(_async_chunk_generator())

    def complete(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        max_continues: int = 5,
        ensure_complete: bool = True,
        continue_prompt: str | Callable = "continue",
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        **params: Any,
    ) -> ChatResult:
        """
        Ensure a complete response, automatically handling truncation.

        **Behavior**: Automatically continues generation if the response is truncated,
        ensuring the returned result is complete (or raises an exception).

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        **History Management**:
        - If `history` is provided, uses it (for multi-turn conversations)
        - If `history` is None, creates a new history internally (for single-turn conversations)
        - The history is automatically updated with the prompt and response

        Use this when:
        - You need a complete response (e.g., JSON extraction)
        - You cannot accept partial responses
        - Reliability is more important than performance

        For single responses (even if truncated), use `chat()` instead.

        Args:
            messages: Input messages.
            history: Optional ChatHistory instance. If None, creates a new one internally.
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues. If False, returns partial result.
            continue_prompt: User prompt for continuation requests. Can be a string or
                a callable with signature: (count: int, max_count: int, current_text: str, original_prompt: str) -> str
            on_progress: Optional progress callback function with signature:
                (count: int, max_count: int, current_result: ChatResult, all_results: List[ChatResult]) -> None
            continue_delay: Delay between continue requests (seconds). Can be a float (fixed delay)
                or tuple (min, max) for random delay. Delay is only applied after the first continue.
            on_error: Error handling strategy: "raise" (default) or "return_partial".
            on_error_callback: Optional error callback function with signature:
                (error: Exception, partial_result: ChatResult) -> dict
            params: Additional parameters to pass to chat and continue requests.

        Returns:
            Complete ChatResult (never truncated, unless max_continues exceeded).

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.

        Examples:
            Single-turn conversation (no history needed):
            >>> result = chat.complete("Write a long JSON", max_tokens=100)
            >>> import json
            >>> json_data = json.loads(result.text)  # Response is complete

            Multi-turn conversation (provide history):
            >>> history = ChatHistory()
            >>> result1 = chat.complete("First question", history=history)
            >>> result2 = chat.complete("Follow-up question", history=history)

            With progress tracking:
            >>> def on_progress(count, max_count, current, all_results):
            ...     print(f"Continuing generation {count}/{max_count}...")
            >>> result = chat.complete("Write JSON", on_progress=on_progress)
        """
        return self._continuer.complete(
            messages=messages,
            history=history,
            max_continues=max_continues,
            ensure_complete=ensure_complete,
            continue_prompt=continue_prompt,
            on_progress=on_progress,
            continue_delay=continue_delay,
            on_error=on_error,
            on_error_callback=on_error_callback,
            **params,
        )

    def complete_stream(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        max_continues: int = 5,
        ensure_complete: bool = True,
        continue_prompt: str | Callable = "continue",
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        **params: Any,
    ) -> StreamingIterator:
        """
        Stream a complete response, automatically handling truncation.

        **Behavior**: Automatically continues streaming if the response is truncated,
        ensuring the final result is complete (or raises an exception).

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        **History Management**:
        - If `history` is provided, uses it (for multi-turn conversations)
        - If `history` is None, creates a new history internally (for single-turn conversations)
        - The history is automatically updated with the prompt and response

        Use this when:
        - You need a complete response with real-time output
        - You cannot accept partial responses
        - You want both streaming and completeness

        For single streaming responses (even if truncated), use `chat.stream()` instead.

        Args:
            messages: Input messages.
            history: Optional ChatHistory instance. If None, creates a new one internally.
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues. If False, returns partial result.
            continue_prompt: User prompt for continuation requests. Can be a string or
                a callable with signature: (count: int, max_count: int, current_text: str, original_prompt: str) -> str
            on_progress: Optional progress callback function with signature:
                (count: int, max_count: int, current_result: ChatResult, all_results: List[ChatResult]) -> None
            continue_delay: Delay between continue requests (seconds). Can be a float (fixed delay)
                or tuple (min, max) for random delay. Delay is only applied after the first continue.
            on_error: Error handling strategy: "raise" (default) or "return_partial".
            on_error_callback: Optional error callback function with signature:
                (error: Exception, partial_result: ChatResult) -> dict
            params: Additional parameters to pass to chat and continue requests.

        Returns:
            StreamingIterator: Iterator that yields ChatStreamChunk objects from
                initial request and all continue requests. Access accumulated result
                via iterator.result.

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.

        Examples:
            Single-turn conversation (no history needed):
            >>> iterator = chat.complete_stream("Write a long JSON", max_tokens=100)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="", flush=True)
            >>> result = iterator.result.to_chat_result()
            >>> import json
            >>> json_data = json.loads(result.text)  # Response is complete

            Multi-turn conversation (provide history):
            >>> history = ChatHistory()
            >>> iterator1 = chat.complete_stream("First question", history=history)
            >>> iterator2 = chat.complete_stream("Follow-up", history=history)
        """
        return self._continuer.complete_stream(
            messages=messages,
            history=history,
            max_continues=max_continues,
            ensure_complete=ensure_complete,
            continue_prompt=continue_prompt,
            on_progress=on_progress,
            continue_delay=continue_delay,
            on_error=on_error,
            on_error_callback=on_error_callback,
            **params,
        )

    async def acomplete(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        max_continues: int = 5,
        ensure_complete: bool = True,
        continue_prompt: str | Callable = "continue",
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        **params: Any,
    ) -> ChatResult:
        """
        Async version of complete().

        Ensure a complete response asynchronously, automatically handling truncation.

        **Behavior**: Automatically continues generation if the response is truncated,
        ensuring the returned result is complete (or raises an exception).

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        Args:
            messages: Input messages.
            history: Optional ChatHistory instance.
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues.
            continue_prompt: User prompt for continuation requests.
            on_progress: Optional progress callback function.
            continue_delay: Delay between continue requests (seconds).
            on_error: Error handling strategy: "raise" (default) or "return_partial".
            on_error_callback: Optional error callback function.
            params: Additional parameters to pass to chat and continue requests.

        Returns:
            Complete ChatResult (never truncated, unless max_continues exceeded).

        Examples:
            >>> result = await chat.acomplete("Write a long JSON", max_tokens=100)
            >>> import json
            >>> json_data = json.loads(result.text)  # Response is complete
        """
        return await self._continuer.acomplete(
            messages=messages,
            history=history,
            max_continues=max_continues,
            ensure_complete=ensure_complete,
            continue_prompt=continue_prompt,
            on_progress=on_progress,
            continue_delay=continue_delay,
            on_error=on_error,
            on_error_callback=on_error_callback,
            **params,
        )

    async def acomplete_stream(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        max_continues: int = 5,
        ensure_complete: bool = True,
        continue_prompt: str | Callable = "continue",
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        **params: Any,
    ) -> AsyncStreamingIterator:
        """
        Async version of complete_stream().

        Stream a complete response asynchronously, automatically handling truncation.

        **Behavior**: Automatically continues streaming if the response is truncated,
        ensuring the final result is complete (or raises an exception).

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        Args:
            messages: Input messages.
            history: Optional ChatHistory instance.
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues.
            continue_prompt: User prompt for continuation requests.
            on_progress: Optional progress callback function.
            continue_delay: Delay between continue requests (seconds).
            on_error: Error handling strategy: "raise" (default) or "return_partial".
            on_error_callback: Optional error callback function.
            params: Additional parameters to pass to chat and continue requests.

        Returns:
            AsyncStreamingIterator: Async iterator that yields ChatStreamChunk objects.

        Examples:
            >>> async for chunk in await chat.acomplete_stream("Write JSON"):
            ...     print(chunk.delta, end="", flush=True)
            >>> result = iterator.result.to_chat_result()
        """
        return await self._continuer.acomplete_stream(
            messages=messages,
            history=history,
            max_continues=max_continues,
            ensure_complete=ensure_complete,
            continue_prompt=continue_prompt,
            on_progress=on_progress,
            continue_delay=continue_delay,
            on_error=on_error,
            on_error_callback=on_error_callback,
            **params,
        )

    def chat_with_history(
        self,
        history: ChatHistory,
        message: str | dict | None = None,
        **params,
    ) -> ChatResult:
        r"""
        Make a chat completion request using history.

        This is a convenience method. You can also use:
        >>> chat(message, history=history, \*\*params)

        Args:
            history: ChatHistory instance to use.
            message: Optional new message to add. If None, uses history as-is.
            ``**params``: Additional parameters to pass to __call__.

        Returns:
            ChatResult from the API call.

        Examples:
            >>> history = ChatHistory.from_messages("Hello")
            >>> result = chat.chat_with_history(history, temperature=0.7)
            >>> # Or with a new message:
            >>> result = chat.chat_with_history(history, "Continue", temperature=0.7)
        """
        if message is not None:
            return self(message, history=history, **params)
        else:
            # Use last user message from history as the message
            last_user = history.get_last_user_message()
            if last_user is None:
                raise ValueError(
                    "History has no user messages. Provide a message parameter."
                )
            return self(last_user, history=history, **params)

    def stream_with_history(
        self,
        history: ChatHistory,
        message: str | dict | None = None,
        **params,
    ) -> StreamingIterator:
        r"""
        Make a streaming chat completion request using history.

        This is a convenience method. You can also use:
        >>> chat.stream(message, history=history, \*\*params)

        Args:
            history: ChatHistory instance to use.
            message: Optional new message to add. If None, uses history as-is.
            ``**params``: Additional parameters to pass to stream().

        Returns:
            StreamingIterator for the streaming response.

        Examples:
            >>> history = ChatHistory.from_messages("Hello")
            >>> iterator = chat.stream_with_history(history, temperature=0.7)
            >>> # Or with a new message:
            >>> iterator = chat.stream_with_history(history, "Continue", temperature=0.7)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="")
        """
        if message is not None:
            return self.stream(message, history=history, **params)
        else:
            # Use last user message from history as the message
            last_user = history.get_last_user_message()
            if last_user is None:
                raise ValueError(
                    "History has no user messages. Provide a message parameter."
                )
            return self.stream(last_user, history=history, **params)
