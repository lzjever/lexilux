"""
Chat API client.

Provides a simple, function-like API for chat completions with support for
both non-streaming and streaming responses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import requests

from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike
from lexilux.chat.params import ChatParams
from lexilux.chat.streaming import StreamingIterator, StreamingResult
from lexilux.chat.utils import normalize_finish_reason, normalize_messages, parse_usage
from lexilux.usage import Json, Usage


class Chat:
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
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ):
        """
        Initialize Chat client.

        Args:
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1").
            api_key: API key for authentication (optional if provided in headers).
            model: Default model to use (can be overridden in __call__).
            timeout_s: Request timeout in seconds.
            headers: Additional headers to include in requests.
            proxies: Optional proxy configuration dict (e.g., {"http": "http://proxy:port"}).
                    If None, uses environment variables (HTTP_PROXY, HTTPS_PROXY).
                    To disable proxies, pass {}.
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
        params: ChatParams | None = None,
        extra: Json | None = None,
        return_raw: bool = False,
    ) -> ChatResult:
        """
        Make a single chat completion request.

        **Behavior**: Returns the response from a single API call, even if truncated.
        Does NOT automatically continue if the response is cut off.

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        Use this when:
        - You accept partial responses
        - You want to handle truncation manually
        - Performance is more important than completeness

        For complete responses, use `chat.complete()` instead.

        Supports both direct parameter passing (backward compatible) and ChatParams
        dataclass for structured configuration.

        Args:
            messages: Messages in various formats (str, list of str, list of dict).
            history: Optional ChatHistory instance. If provided, history.messages are
                prepended to messages, and a clone is automatically updated with the
                user message and assistant response after successful completion.
                The original history is never modified.
            model: Model to use (overrides default).
            system: Optional system message.
            temperature: Sampling temperature (0.0-2.0). Higher values make output
                more random, lower values more focused. Default: 0.7
            top_p: Nucleus sampling parameter (0.0-1.0). Alternative to temperature.
                Default: 1.0
            max_tokens: Maximum tokens to generate. Default: None (no limit)
            stop: Stop sequences (str or list of str). API stops at these sequences.
            presence_penalty: Penalty for new topics (-2.0 to 2.0). Positive values
                encourage new topics. Default: 0.0
            frequency_penalty: Penalty for repetition (-2.0 to 2.0). Positive values
                reduce repetition. Default: 0.0
            logit_bias: Modify token likelihood. Dict mapping token IDs to bias
                values (-100 to 100). Default: None
            user: Unique identifier for end-user (for monitoring/rate limiting).
            n: Number of chat completion choices to generate. Default: 1
            params: ChatParams dataclass instance. If provided, overrides individual
                parameters above. Useful for structured configuration.
            extra: Additional custom parameters for non-standard providers.
                Merged with params if both are provided.
            return_raw: Whether to include full raw response.

        Returns:
            ChatResult with text and usage. May be truncated if finish_reason == "length".

        Examples:
            Basic usage (may be truncated):
            >>> result = chat("Hello", temperature=0.5, max_tokens=100)
            >>> if result.finish_reason == "length":
            ...     print("Response was truncated")

            With explicit history (immutable):
            >>> history = ChatHistory()
            >>> result = chat("Hello", history=history)
            >>> # Original history is not modified, working copy is used internally

        Raises:
            requests.RequestException: On network or HTTP errors (connection timeout,
                connection reset, DNS resolution failure, etc.). When this exception
                is raised during streaming, the iterator will stop and no more chunks
                will be yielded. If the stream was interrupted before receiving a
                done=True chunk, finish_reason will not be available. This indicates
                a network/connection problem, not a normal completion.
            ValueError: On invalid input or response format.
        """
        # Normalize messages
        normalized_messages = normalize_messages(messages, system=system)

        # If history is provided, create working copy (immutable) and prepend history messages
        user_messages_to_add: list[str] = []
        working_history: ChatHistory | None = None
        if history is not None:
            # Create working history (immutable - clone original)
            working_history = history.clone()
            # Prepend history messages
            history_messages = working_history.get_messages(include_system=True)
            normalized_messages = history_messages + normalized_messages
            # Extract new user messages for history update
            for msg in normalize_messages(messages, system=system):
                if msg.get("role") == "user":
                    user_messages_to_add.append(msg.get("content", ""))

        # Prepare request
        model = model or self.model
        if not model:
            raise ValueError("Model must be specified (either in __init__ or __call__)")

        # Build parameters from ChatParams or individual args
        if params is not None:
            # Use ChatParams as base, override with individual args if provided
            param_dict = params.to_dict(exclude_none=True)
            # Override with explicit parameters if provided
            if temperature is not None:
                param_dict["temperature"] = temperature
            if top_p is not None:
                param_dict["top_p"] = top_p
            if max_tokens is not None:
                param_dict["max_tokens"] = max_tokens
            if stop is not None:
                if isinstance(stop, str):
                    param_dict["stop"] = [stop]
                else:
                    param_dict["stop"] = list(stop)
            if presence_penalty is not None:
                param_dict["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                param_dict["frequency_penalty"] = frequency_penalty
            if logit_bias is not None:
                param_dict["logit_bias"] = logit_bias
            if user is not None:
                param_dict["user"] = user
            if n is not None:
                param_dict["n"] = n
        else:
            # Build from individual parameters (backward compatible)
            param_dict: Json = {}
            if temperature is not None:
                param_dict["temperature"] = temperature
            else:
                param_dict["temperature"] = 0.7  # Default
            if top_p is not None:
                param_dict["top_p"] = top_p
            if max_tokens is not None:
                param_dict["max_tokens"] = max_tokens
            if stop is not None:
                if isinstance(stop, str):
                    param_dict["stop"] = [stop]
                else:
                    param_dict["stop"] = list(stop)
            if presence_penalty is not None:
                param_dict["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                param_dict["frequency_penalty"] = frequency_penalty
            if logit_bias is not None:
                param_dict["logit_bias"] = logit_bias
            if user is not None:
                param_dict["user"] = user
            if n is not None:
                param_dict["n"] = n

        # Build payload
        payload: Json = {
            "model": model,
            "messages": normalized_messages,
            **param_dict,
        }

        # Merge extra parameters (highest priority)
        if extra:
            payload.update(extra)

        # Update working history BEFORE request (add user messages)
        # This ensures user messages are recorded even if request fails
        # Note: working_history is a clone, original history is never modified
        if working_history is not None:
            for user_msg in user_messages_to_add:
                working_history.add_user(user_msg)

        # Make request (may raise exception)
        url = f"{self.base_url}/chat/completions"
        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_s,
            proxies=self.proxies,
        )
        response.raise_for_status()

        response_data = response.json()

        # Parse response
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in API response")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError(f"Invalid choice format: expected dict, got {type(choice)}")

        # Extract text content
        message = choice.get("message", {})
        if not isinstance(message, dict):
            message = {}
        text = message.get("content", "") or ""

        # Normalize finish_reason (defensive against invalid implementations)
        finish_reason = normalize_finish_reason(choice.get("finish_reason"))

        # Parse usage
        usage = parse_usage(response_data)

        # Create result
        result = ChatResult(
            text=text,
            usage=usage,
            finish_reason=finish_reason,
            raw=response_data if return_raw else {},
        )

        # Add assistant response to working history ONLY on success (after all exceptions are handled)
        # Note: working_history is a clone, original history is never modified
        if working_history is not None:
            working_history.append_result(result)

        return result

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
        params: ChatParams | None = None,
        extra: Json | None = None,
        include_usage: bool = True,
        return_raw_events: bool = False,
    ) -> StreamingIterator:
        """
        Stream a single chat completion response.

        **Behavior**: Streams the response from a single API call, even if truncated.
        Does NOT automatically continue if the response is cut off.

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        Use this when:
        - You want real-time output
        - You accept partial responses
        - You want to handle truncation manually

        For complete streaming responses, use `chat.complete_stream()` instead.

        Supports both direct parameter passing (backward compatible) and ChatParams
        dataclass for structured configuration.

        Args:
            messages: Messages in various formats.
            history: Optional ChatHistory instance. If provided, history.messages are
                prepended to messages, and a clone is automatically updated with the
                user message and assistant response during streaming.
                The original history is never modified.
            model: Model to use (overrides default).
            system: Optional system message.
            temperature: Sampling temperature (0.0-2.0). Higher values make output
                more random, lower values more focused. Default: 0.7
            top_p: Nucleus sampling parameter (0.0-1.0). Alternative to temperature.
                Default: 1.0
            max_tokens: Maximum tokens to generate. Default: None (no limit)
            stop: Stop sequences (str or list of str). API stops at these sequences.
            presence_penalty: Penalty for new topics (-2.0 to 2.0). Positive values
                encourage new topics. Default: 0.0
            frequency_penalty: Penalty for repetition (-2.0 to 2.0). Positive values
                reduce repetition. Default: 0.0
            logit_bias: Modify token likelihood. Dict mapping token IDs to bias
                values (-100 to 100). Default: None
            user: Unique identifier for end-user (for monitoring/rate limiting).
            params: ChatParams dataclass instance. If provided, overrides individual
                parameters above. Useful for structured configuration.
            extra: Additional custom parameters for non-standard providers.
                Merged with params if both are provided.
            include_usage: Whether to request usage in the final chunk (OpenAI-style).
            return_raw_events: Whether to include raw event data in chunks.

        Returns:
            StreamingIterator: Iterator that yields ChatStreamChunk objects.
                    Access accumulated result via iterator.result.

        Raises:
            requests.RequestException: On network or HTTP errors (connection timeout,
                connection reset, DNS resolution failure, etc.). When this exception
                is raised during streaming, the iterator will stop and no more chunks
                will be yielded. If the stream was interrupted before receiving a
                done=True chunk, finish_reason will not be available. This indicates
                a network/connection problem, not a normal completion.
            ValueError: On invalid input or response format.

        Examples:
            Basic streaming (may be truncated):
            >>> for chunk in chat.stream("Hello", temperature=0.5):
            ...     print(chunk.delta, end="")
            >>> result = iterator.result.to_chat_result()
            >>> if result.finish_reason == "length":
            ...     print("Response was truncated")
        """
        # Normalize messages
        normalized_messages = normalize_messages(messages, system=system)

        # If history is provided, create working copy (immutable) and prepend history messages
        user_messages_to_add: list[str] = []
        working_history: ChatHistory | None = None
        if history is not None:
            # Create working history (immutable - clone original)
            working_history = history.clone()
            # Prepend history messages
            history_messages = working_history.get_messages(include_system=True)
            normalized_messages = history_messages + normalized_messages
            # Extract new user messages for history update
            for msg in normalize_messages(messages, system=system):
                if msg.get("role") == "user":
                    user_messages_to_add.append(msg.get("content", ""))

        # Prepare request
        model = model or self.model
        if not model:
            raise ValueError("Model must be specified (either in __init__ or __call__)")

        # Build parameters from ChatParams or individual args
        if params is not None:
            # Use ChatParams as base, override with individual args if provided
            param_dict = params.to_dict(exclude_none=True)
            # Override with explicit parameters if provided
            if temperature is not None:
                param_dict["temperature"] = temperature
            if top_p is not None:
                param_dict["top_p"] = top_p
            if max_tokens is not None:
                param_dict["max_tokens"] = max_tokens
            if stop is not None:
                if isinstance(stop, str):
                    param_dict["stop"] = [stop]
                else:
                    param_dict["stop"] = list(stop)
            if presence_penalty is not None:
                param_dict["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                param_dict["frequency_penalty"] = frequency_penalty
            if logit_bias is not None:
                param_dict["logit_bias"] = logit_bias
            if user is not None:
                param_dict["user"] = user
        else:
            # Build from individual parameters (backward compatible)
            param_dict: Json = {}
            if temperature is not None:
                param_dict["temperature"] = temperature
            else:
                param_dict["temperature"] = 0.7  # Default
            if top_p is not None:
                param_dict["top_p"] = top_p
            if max_tokens is not None:
                param_dict["max_tokens"] = max_tokens
            if stop is not None:
                if isinstance(stop, str):
                    param_dict["stop"] = [stop]
                else:
                    param_dict["stop"] = list(stop)
            if presence_penalty is not None:
                param_dict["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                param_dict["frequency_penalty"] = frequency_penalty
            if logit_bias is not None:
                param_dict["logit_bias"] = logit_bias
            if user is not None:
                param_dict["user"] = user

        # Build payload
        payload: Json = {
            "model": model,
            "messages": normalized_messages,
            "stream": True,
            **param_dict,
        }

        if include_usage:
            # OpenAI-style: request usage in final chunk
            payload["stream_options"] = {"include_usage": True}

        # Merge extra parameters (highest priority)
        if extra:
            payload.update(extra)

        # Make streaming request
        url = f"{self.base_url}/chat/completions"
        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_s,
            stream=True,
            proxies=self.proxies,
        )
        response.raise_for_status()

        # Create internal chunk generator
        def _chunk_generator() -> Iterator[ChatStreamChunk]:
            """Internal generator for streaming chunks."""
            accumulated_text = ""
            final_usage: Usage | None = None
            final_finish_reason: str | None = None  # Track finish_reason from chunks

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    # Final chunk with usage (if include_usage=True)
                    # Use finish_reason from previous chunk if available
                    if final_usage is None:
                        # No usage received, create empty usage
                        final_usage = Usage()
                    yield ChatStreamChunk(
                        delta="",
                        done=True,
                        usage=final_usage,
                        finish_reason=final_finish_reason,  # Preserve finish_reason from previous chunk
                        raw={"done": True} if return_raw_events else {},
                    )
                    break

                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Parse event
                choices = event_data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                if not isinstance(choice, dict):
                    # Skip invalid choice format
                    continue

                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}
                content = delta.get("content") or ""

                # Normalize finish_reason (defensive against invalid implementations)
                finish_reason = normalize_finish_reason(choice.get("finish_reason"))
                # done is True when finish_reason is a non-empty string
                done = finish_reason is not None

                # Track finish_reason for [DONE] chunk
                if finish_reason is not None:
                    final_finish_reason = finish_reason

                # Accumulate text
                accumulated_text += content

                # Parse usage if present (usually only in final chunk when include_usage=True)
                usage = None
                if "usage" in event_data:
                    usage = parse_usage(event_data)
                    final_usage = usage
                elif done and final_usage is None:
                    # Final chunk but no usage yet - create empty usage
                    usage = Usage()
                    final_usage = usage
                else:
                    # Intermediate chunk - empty usage
                    usage = Usage()

                yield ChatStreamChunk(
                    delta=content,
                    done=done,
                    usage=usage,
                    finish_reason=finish_reason,
                    raw=event_data if return_raw_events else {},
                )

        # Create StreamingIterator
        chunk_iterator = _chunk_generator()
        streaming_iterator = StreamingIterator(chunk_iterator)

        # If working history is provided, wrap iterator to update working history
        # Note: working_history is a clone, original history is never modified
        if working_history is not None:
            # Add user messages to working history before streaming
            for user_msg in user_messages_to_add:
                working_history.add_user(user_msg)
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
                self._assistant_added = False  # Track if assistant message has been added

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
                (count: int, max_count: int, current_result: ChatResult, all_results: list[ChatResult]) -> None
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
        from lexilux.chat.continue_ import ChatContinue
        from lexilux.chat.exceptions import ChatIncompleteResponseError

        # Create working history (immutable - clone if provided, otherwise create new)
        working_history = history.clone() if history is not None else ChatHistory()

        # Get original prompt for dynamic continue_prompt generation
        original_prompt = messages if isinstance(messages, str) else str(messages)

        # Make initial request
        result = self(messages, history=working_history, **params)

        # If truncated, continue with customizable strategy
        if result.finish_reason == "length":
            try:
                result = ChatContinue.continue_request(
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
                # If on_error="return_partial", the error handler should have returned partial result
                # So if we get here, it means on_error="raise" or error handler raised
                if ensure_complete:
                    raise ChatIncompleteResponseError(
                        f"Failed to get complete response after {max_continues} continues: {e}",
                        final_result=result,
                        continue_count=0,
                        max_continues=max_continues,
                    ) from e
                # If ensure_complete=False, re-raise the exception
                raise

        # Check if still truncated after continues
        # Only raise if ensure_complete=True AND on_error="raise" (not "return_partial")
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
                (count: int, max_count: int, current_result: ChatResult, all_results: list[ChatResult]) -> None
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
            >>> json_data = json.loads(result.text)  # Guaranteed complete

            Multi-turn conversation (provide history):
            >>> history = ChatHistory()
            >>> iterator1 = chat.complete_stream("First question", history=history)
            >>> iterator2 = chat.complete_stream("Follow-up", history=history)
        """
        from lexilux.chat.continue_ import ChatContinue
        from lexilux.chat.exceptions import ChatIncompleteResponseError

        # Create working history (immutable - clone if provided, otherwise create new)
        working_history = history.clone() if history is not None else ChatHistory()

        # Get original prompt for dynamic continue_prompt generation
        original_prompt = messages if isinstance(messages, str) else str(messages)

        # Create generator that yields initial chunks and handles continues
        def _complete_stream_generator() -> Iterator[ChatStreamChunk]:
            """Generator that yields chunks from initial request and continues."""
            # Start with streaming request
            initial_iterator = self.stream(messages, history=working_history, **params)

            # Yield all chunks from initial request
            for chunk in initial_iterator:
                yield chunk

            # Get initial result
            initial_result = initial_iterator.result.to_chat_result()

            # If truncated, continue with streaming
            if initial_result.finish_reason == "length":
                try:
                    continue_iterator = ChatContinue.continue_request_stream(
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
                    # Yield all chunks from continue requests
                    for chunk in continue_iterator:
                        yield chunk
                except Exception as e:
                    if ensure_complete:
                        # Get final result for error
                        final_result = initial_result
                        raise ChatIncompleteResponseError(
                            f"Failed to get complete response after {max_continues} continues: {e}",
                            final_result=final_result,
                            continue_count=0,
                            max_continues=max_continues,
                        ) from e
                    raise

        # Create StreamingIterator with custom result and error checking
        class CompleteStreamingIterator(StreamingIterator):
            """Iterator for complete_stream with merged result and error checking."""

            def __init__(
                self,
                chunk_gen: Iterator[ChatStreamChunk],
                max_continues: int,
                ensure_complete: bool,
            ):
                super().__init__(chunk_gen)
                self._max_continues = max_continues
                self._ensure_complete = ensure_complete
                self._iterated = False

            def __iter__(self) -> Iterator[ChatStreamChunk]:
                """Iterate chunks and check for errors after iteration."""
                self._iterated = True
                for chunk in self._iterator:
                    self._result.update(chunk)
                    yield chunk

                # After iteration, check if we need to raise error
                if self._ensure_complete:
                    final_result = self.result.to_chat_result()
                    if final_result.finish_reason == "length":
                        from lexilux.chat.exceptions import ChatIncompleteResponseError

                        raise ChatIncompleteResponseError(
                            f"Response still truncated after {self._max_continues} continues. "
                            f"Consider increasing max_continues or max_tokens.",
                            final_result=final_result,
                            continue_count=self._max_continues,
                            max_continues=self._max_continues,
                        )

        return CompleteStreamingIterator(
            _complete_stream_generator(),
            max_continues,
            ensure_complete,
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
                raise ValueError("History has no user messages. Provide a message parameter.")
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
                raise ValueError("History has no user messages. Provide a message parameter.")
            return self.stream(last_user, history=history, **params)
