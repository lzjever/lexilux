"""
Chat API client.

Provides a simple, function-like API for chat completions with support for
both non-streaming and streaming responses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from lexilux._base import BaseAPIClient
from lexilux.chat._request import (
    SSEChatStreamParser,
    build_params_dict,
    build_payload,
    parse_chat_completion_response,
    prepare_messages_for_request,
)
from lexilux.chat.conversation import Conversation
from lexilux.chat.exceptions import ChatIncompleteResponseError
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike
from lexilux.chat.params import ChatParams
from lexilux.chat.streaming import (
    AsyncStreamingIterator,
    StreamingIterator,
    StreamingResult,
)
from lexilux.usage import Json

if TYPE_CHECKING:
    from lexilux.chat.tools import Tool


def _get_working_history(history: ChatHistory | None) -> ChatHistory:
    return history.clone() if history is not None else ChatHistory()


def _get_original_prompt(messages: MessagesLike) -> str:
    return messages if isinstance(messages, str) else str(messages)


class _CompleteStreamingIterator(StreamingIterator):
    """StreamingIterator wrapper that checks for truncation after iteration."""

    def __init__(
        self,
        chunk_gen: Iterator[ChatStreamChunk],
        max_continues: int,
        ensure_complete: bool,
    ):
        super().__init__(chunk_gen)
        self._max_continues = max_continues
        self._ensure_complete = ensure_complete

    def __iter__(self) -> Iterator[ChatStreamChunk]:
        for chunk in self._iterator:
            self._result.update(chunk)
            yield chunk

        if self._ensure_complete:
            final_result = self.result.to_chat_result()
            if final_result.finish_reason == "length":
                raise ChatIncompleteResponseError(
                    f"Response still truncated after {self._max_continues} continues. "
                    f"Consider increasing max_continues or max_tokens.",
                    final_result=final_result,
                    continue_count=self._max_continues,
                    max_continues=self._max_continues,
                )


class _AsyncCompleteStreamingIterator(AsyncStreamingIterator):
    """AsyncStreamingIterator wrapper that checks for truncation on completion."""

    def __init__(
        self,
        chunk_gen: AsyncIterator[ChatStreamChunk],
        max_continues: int,
        ensure_complete: bool,
    ):
        super().__init__(chunk_gen)
        self._max_continues = max_continues
        self._ensure_complete = ensure_complete

    async def __anext__(self) -> ChatStreamChunk:
        try:
            chunk = await self._iterator.__anext__()
            self._result.update(chunk)
            return chunk
        except StopAsyncIteration:
            if self._ensure_complete:
                final_result = self.result.to_chat_result()
                if final_result.finish_reason == "length":
                    raise ChatIncompleteResponseError(
                        f"Response still truncated after {self._max_continues} continues.",
                        final_result=final_result,
                        continue_count=self._max_continues,
                        max_continues=self._max_continues,
                    )
            raise


class Chat(BaseAPIClient):
    """
    Chat API client.

    Provides a simple, function-like API for chat completions with support for
    both non-streaming and streaming responses.

    Examples:
        >>> chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")
        >>> result = chat("Hello, world!")
        >>> print(result.text)

        >>> # Streaming
        >>> for chunk in chat.stream("Tell me a joke"):
        ...     print(chunk.delta, end="")

        >>> # With system message
        >>> result = chat("What is Python?", system="You are a helpful assistant")
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
        pool_connections: int = 10,
        pool_maxsize: int = 10,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
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
            pool_connections: Number of connection pools to cache (default: 10).
            pool_maxsize: Maximum number of connections in pool (default: 10).
            headers: Additional headers to include in requests.
            proxies: Optional proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.

        Note:
            Connection pooling and retry logic are handled by BaseAPIClient.
            Set max_retries > 0 to enable automatic retries on transient failures.
        """
        # Initialize base client with connection pooling and retry support
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            max_retries=max_retries,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            headers=headers,
            proxies=proxies,
        )

        # Chat-specific attributes
        self.model = model

    @property
    def timeout_s(self) -> float:
        """
        Backward compatibility property for timeout.

        Returns the timeout value (or read timeout if tuple).
        """
        if isinstance(self.timeout, tuple):
            return self.timeout[1]  # Return read timeout
        return self.timeout

    def _prepare_chat_request(
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
        **kwargs: Any,
    ) -> tuple[Json, ChatHistory | None]:
        """Prepares the request payload and working history."""
        normalized_messages, working_history, user_messages_to_add = (
            prepare_messages_for_request(
                messages,
                system=system,
                history=history,
            )
        )

        final_model = model or self.model
        if not final_model:
            raise ValueError("Model must be specified (either in __init__ or in call)")

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

    def _process_chat_response(
        self,
        response_data: Json,
        working_history: ChatHistory | None,
        return_raw: bool,
    ) -> ChatResult:
        """Processes the JSON response and updates history."""
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
        return_raw: bool = False,
    ) -> ChatResult:
        """
        Make a single chat completion request.
        """
        payload, working_history = self._prepare_chat_request(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=False,
            include_usage=False,
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
        response_data = response.json()
        return self._process_chat_response(
            response_data, working_history, return_raw=return_raw
        )

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
        include_usage: bool = True,
        return_raw_events: bool = False,
        # ✅ NEW: Add reasoning parameter
        include_reasoning: bool = False,
    ) -> StreamingIterator:
        """
        Stream a single chat completion response.
        """
        payload, working_history = self._prepare_chat_request(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=True,
            include_usage=include_usage,
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

        # Make streaming request
        response = self._make_streaming_request("chat/completions", payload)

        # Create internal chunk generator
        def _chunk_generator() -> Iterator[ChatStreamChunk]:
            """Internal generator for streaming chunks."""
            parser = SSEChatStreamParser(
                return_raw_events=return_raw_events,
                include_reasoning=include_reasoning,  # ✅ NEW
            )
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

        # Create StreamingIterator
        chunk_iterator = _chunk_generator()
        streaming_iterator = StreamingIterator(chunk_iterator)

        # If working history is provided, wrap iterator to update working history
        if working_history is not None:
            streaming_iterator = self._wrap_streaming_with_history(
                streaming_iterator, working_history
            )

        return streaming_iterator

    def _wrap_streaming_with_history(
        self,
        iterator: StreamingIterator,
        history: ChatHistory,
    ) -> StreamingIterator:
        """
        Wrap streaming iterator to automatically update history.

        Behavior:
        - User messages should already be added to history before calling this method
        - Assistant message is added to history only on first iteration (lazy initialization)
        - Assistant message content is updated on each iteration with accumulated text
        - If iterator is never iterated, no assistant message is added

        Args:
            iterator: StreamingIterator to wrap.
            history: ChatHistory instance to update.

        Returns:
            Wrapped StreamingIterator that updates history on each chunk.
        """

        # Wrap iterator to update history
        class HistoryUpdatingIterator(StreamingIterator):
            """Iterator wrapper that updates history on each chunk."""

            def __init__(self, base_iterator: StreamingIterator, history: ChatHistory):
                # Initialize with base iterator's internal iterator
                super().__init__(base_iterator._iterator)
                self._base = base_iterator
                self._history = history
                # Use base iterator's result (which is already accumulating)
                self._result = base_iterator.result
                self._assistant_added = (
                    False  # Track if assistant message has been added
                )

            def __iter__(self) -> Iterator[ChatStreamChunk]:
                """Iterate chunks and update history."""
                for chunk in self._base:
                    # Add assistant message on first iteration (lazy initialization)
                    if not self._assistant_added:
                        self._history.add_assistant("")
                        self._assistant_added = True

                    # Update history's last assistant message with current accumulated text
                    if (
                        self._history.messages
                        and self._history.messages[-1].get("role") == "assistant"
                    ):
                        self._history.messages[-1]["content"] = self.result.text
                    yield chunk

            @property
            def result(self) -> StreamingResult:
                """Get accumulated result."""
                return self._result

        return HistoryUpdatingIterator(iterator, history)

    # =========================================================================
    # Async Methods
    # =========================================================================

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
        return_raw: bool = False,
    ) -> ChatResult:
        """
        Make an async chat completion request.
        """
        payload, working_history = self._prepare_chat_request(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=False,
            include_usage=False,
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
        response_data = response.json()
        return self._process_chat_response(
            response_data, working_history, return_raw=return_raw
        )

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
        include_usage: bool = True,
        return_raw_events: bool = False,
        # ✅ NEW: Add reasoning parameter
        include_reasoning: bool = False,
    ) -> AsyncStreamingIterator:
        """
        Stream an async chat completion response.
        """
        payload, working_history = self._prepare_chat_request(
            messages=messages,
            history=history,
            model=model,
            system=system,
            params=params,
            extra=extra,
            stream=True,
            include_usage=include_usage,
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
            parser = SSEChatStreamParser(
                return_raw_events=return_raw_events,
                include_reasoning=include_reasoning,  # ✅ NEW
            )
            async for line in self._amake_streaming_request(
                "chat/completions", payload
            ):
                chunk = parser.feed_line(line)
                if chunk is None:
                    continue
                yield chunk
                if parser.done:
                    break

        # Create AsyncStreamingIterator
        chunk_iterator = _async_chunk_generator()
        streaming_iterator = AsyncStreamingIterator(chunk_iterator)

        # Wrap with history updating if needed
        if working_history is not None:
            streaming_iterator = self._wrap_async_streaming_with_history(
                streaming_iterator, working_history
            )

        return streaming_iterator

    def _wrap_async_streaming_with_history(
        self,
        iterator: AsyncStreamingIterator,
        history: ChatHistory,
    ) -> AsyncStreamingIterator:
        """
        Wrap async streaming iterator to automatically update history.

        Args:
            iterator: AsyncStreamingIterator to wrap.
            history: ChatHistory instance to update.

        Returns:
            Wrapped AsyncStreamingIterator that updates history on each chunk.
        """

        class AsyncHistoryUpdatingIterator(AsyncStreamingIterator):
            """Async iterator wrapper that updates history on each chunk."""

            def __init__(
                self, base_iterator: AsyncStreamingIterator, history: ChatHistory
            ):
                super().__init__(base_iterator._iterator)
                self._base = base_iterator
                self._history = history
                self._result = base_iterator.result
                self._assistant_added = False

            def __aiter__(self) -> AsyncIterator[ChatStreamChunk]:
                return self

            async def __anext__(self) -> ChatStreamChunk:
                try:
                    chunk = await self._base.__anext__()

                    # Add assistant message on first iteration
                    if not self._assistant_added:
                        self._history.add_assistant("")
                        self._assistant_added = True

                    # Update history's last assistant message
                    if (
                        self._history.messages
                        and self._history.messages[-1].get("role") == "assistant"
                    ):
                        self._history.messages[-1]["content"] = self.result.text

                    return chunk
                except StopAsyncIteration:
                    raise

            @property
            def result(self) -> StreamingResult:
                return self._result

        return AsyncHistoryUpdatingIterator(iterator, history)

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
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            Complete ChatResult (never truncated, unless max_continues exceeded).

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.

        Examples:
            Single-turn conversation (no history needed):
            >>> result = chat.complete("Write a long JSON", max_tokens=100)
            >>> import json
            >>> json_data = json.loads(result.text)  # Guaranteed complete

            Multi-turn conversation (provide history):
            >>> history = ChatHistory()
            >>> result1 = chat.complete("First question", history=history)
            >>> result2 = chat.complete("Follow-up question", history=history)

            With progress tracking:
            >>> def on_progress(count, max_count, current, all_results):
            ...     print(f"继续生成 {count}/{max_count}...")
            >>> result = chat.complete("Write JSON", on_progress=on_progress)
        """
        working_history = _get_working_history(history)
        original_prompt = _get_original_prompt(messages)

        result = self(messages, history=working_history, **params)
        if result.finish_reason == "length":
            try:
                result = Conversation.continue_request(
                    self,
                    result,
                    history=working_history,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except Exception as e:
                if ensure_complete:
                    raise ChatIncompleteResponseError(
                        f"Failed to get complete response after {max_continues} continues: {e}",
                        final_result=result,
                        continue_count=0,
                        max_continues=max_continues,
                    ) from e
                raise

        if ensure_complete and result.finish_reason == "length" and on_error == "raise":
            raise ChatIncompleteResponseError(
                f"Response still truncated after {max_continues} continues. "
                f"Consider increasing max_continues or max_tokens.",
                final_result=result,
                continue_count=max_continues,
                max_continues=max_continues,
            )

        return result

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
            **params: Additional parameters to pass to chat and continue requests.

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
            >>> json_data = json.loads(result.text)  # Guaranteed complete

            Multi-turn conversation (provide history):
            >>> history = ChatHistory()
            >>> iterator1 = chat.complete_stream("First question", history=history)
            >>> iterator2 = chat.complete_stream("Follow-up", history=history)
        """
        working_history = _get_working_history(history)
        original_prompt = _get_original_prompt(messages)

        def _complete_stream_generator() -> Iterator[ChatStreamChunk]:
            initial_iterator = self.stream(messages, history=working_history, **params)
            yield from initial_iterator

            initial_result = initial_iterator.result.to_chat_result()
            if initial_result.finish_reason != "length":
                return

            try:
                continue_iterator = Conversation.continue_request_stream(
                    self,
                    initial_result,
                    history=working_history,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except Exception as e:
                if ensure_complete:
                    raise ChatIncompleteResponseError(
                        f"Failed to get complete response after {max_continues} continues: {e}",
                        final_result=initial_result,
                        continue_count=0,
                        max_continues=max_continues,
                    ) from e
                raise

            yield from continue_iterator

        return _CompleteStreamingIterator(
            _complete_stream_generator(),
            max_continues=max_continues,
            ensure_complete=ensure_complete,
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
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            Complete ChatResult (never truncated, unless max_continues exceeded).

        Examples:
            >>> result = await chat.acomplete("Write a long JSON", max_tokens=100)
            >>> import json
            >>> json_data = json.loads(result.text)  # Guaranteed complete

            Concurrent complete requests:
            >>> tasks = [chat.acomplete(f"Write story {i}") for i in range(3)]
            >>> results = await asyncio.gather(*tasks)
        """
        working_history = _get_working_history(history)
        original_prompt = _get_original_prompt(messages)

        result = await self.acall(messages, history=working_history, **params)
        if result.finish_reason == "length":
            try:
                result = await Conversation.acontinue_request(
                    self,
                    result,
                    history=working_history,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except Exception as e:
                if ensure_complete:
                    raise ChatIncompleteResponseError(
                        f"Failed to get complete response after {max_continues} continues: {e}",
                        final_result=result,
                        continue_count=0,
                        max_continues=max_continues,
                    ) from e
                raise

        if ensure_complete and result.finish_reason == "length" and on_error == "raise":
            raise ChatIncompleteResponseError(
                f"Response still truncated after {max_continues} continues. "
                f"Consider increasing max_continues or max_tokens.",
                final_result=result,
                continue_count=max_continues,
                max_continues=max_continues,
            )

        return result

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
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            AsyncStreamingIterator: Async iterator that yields ChatStreamChunk objects.

        Examples:
            >>> async for chunk in await chat.acomplete_stream("Write JSON"):
            ...     print(chunk.delta, end="", flush=True)
            >>> result = iterator.result.to_chat_result()
        """
        working_history = _get_working_history(history)
        original_prompt = _get_original_prompt(messages)

        async def _async_complete_stream_generator() -> AsyncIterator[ChatStreamChunk]:
            initial_iterator = await self.astream(
                messages, history=working_history, **params
            )
            async for chunk in initial_iterator:
                yield chunk

            initial_result = initial_iterator.result.to_chat_result()
            if initial_result.finish_reason != "length":
                return

            try:
                continue_iterator = await Conversation.acontinue_request_stream(
                    self,
                    initial_result,
                    history=working_history,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except Exception as e:
                if ensure_complete:
                    raise ChatIncompleteResponseError(
                        f"Failed to get complete response after {max_continues} continues: {e}",
                        final_result=initial_result,
                        continue_count=0,
                        max_continues=max_continues,
                    ) from e
                raise

            async for chunk in continue_iterator:
                yield chunk

        return _AsyncCompleteStreamingIterator(
            _async_complete_stream_generator(),
            max_continues=max_continues,
            ensure_complete=ensure_complete,
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
