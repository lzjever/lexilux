"""
Chat API client.

Provides a simple, function-like API for chat completions with support for
both non-streaming and streaming responses.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Sequence

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
        Make a non-streaming chat completion request.

        Supports both direct parameter passing (backward compatible) and ChatParams
        dataclass for structured configuration.

        Args:
            messages: Messages in various formats (str, list of str, list of dict).
            history: Optional ChatHistory instance. If provided, history.messages are
                prepended to messages, and history is automatically updated with the
                user message and assistant response after successful completion.
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
            ChatResult with text and usage.

        Raises:
            requests.RequestException: On network or HTTP errors (connection timeout,
                connection reset, DNS resolution failure, etc.). When this exception
                is raised during streaming, the iterator will stop and no more chunks
                will be yielded. If the stream was interrupted before receiving a
                done=True chunk, finish_reason will not be available. This indicates
                a network/connection problem, not a normal completion.
            ValueError: On invalid input or response format.

        Examples:
            Basic usage:
            >>> result = chat("Hello", temperature=0.5, max_tokens=100)

            With explicit history:
            >>> history = ChatHistory()
            >>> result = chat("Hello", history=history)
            >>> # history now contains user: "Hello", assistant: result.text

            Using ChatParams:
            >>> from lexilux import ChatParams
            >>> params = ChatParams(temperature=0.5, max_tokens=100)
            >>> result = chat("Hello", params=params)

            Combining params and extra:
            >>> result = chat("Hello", params=params, extra={"custom": "value"})
        """
        # Normalize messages
        normalized_messages = normalize_messages(messages, system=system)

        # If history is provided, prepend history messages and extract user messages for history update
        user_messages_to_add: list[str] = []
        if history is not None:
            # Prepend history messages
            history_messages = history.get_messages(include_system=True)
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

        # Update history BEFORE request (add user messages)
        # This ensures user messages are recorded even if request fails
        if history is not None:
            for user_msg in user_messages_to_add:
                history.add_user(user_msg)

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

        # Add assistant response to history ONLY on success (after all exceptions are handled)
        if history is not None:
            history.append_result(result)

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
        Make a streaming chat completion request.

        Supports both direct parameter passing (backward compatible) and ChatParams
        dataclass for structured configuration.

        Args:
            messages: Messages in various formats.
            history: Optional ChatHistory instance. If provided, history.messages are
                prepended to messages, and history is automatically updated with the
                user message and assistant response during streaming.
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
            Basic streaming:
            >>> for chunk in chat.stream("Hello", temperature=0.5):
            ...     print(chunk.delta, end="")

            With explicit history:
            >>> history = ChatHistory()
            >>> iterator = chat.stream("Hello", history=history)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="")
            >>> # history now contains user: "Hello", assistant: iterator.result.text

            Using ChatParams:
            >>> from lexilux import ChatParams
            >>> params = ChatParams(temperature=0.5, max_tokens=100)
            >>> for chunk in chat.stream("Hello", params=params):
            ...     print(chunk.delta, end="")
        """
        # Normalize messages
        normalized_messages = normalize_messages(messages, system=system)

        # If history is provided, prepend history messages and extract user messages for history update
        user_messages_to_add: list[str] = []
        if history is not None:
            # Prepend history messages
            history_messages = history.get_messages(include_system=True)
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

        # If history is provided, wrap iterator to update history
        if history is not None:
            # Add user messages to history before streaming
            for user_msg in user_messages_to_add:
                history.add_user(user_msg)
            streaming_iterator = self._wrap_streaming_with_history(streaming_iterator, history)

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

    def continue_if_needed(
        self,
        result: ChatResult,
        *,
        history: ChatHistory,
        max_continues: int = 3,
        continue_prompt: str = "continue",
        **params: Any,
    ) -> ChatResult:
        """
        Continue generation if result is truncated (finish_reason == "length").

        If result.finish_reason != "length", returns result unchanged.
        Otherwise, automatically continues generation until complete or max_continues reached.

        Args:
            result: Previous result to check and potentially continue.
            history: ChatHistory instance (required for continuation).
            max_continues: Maximum number of continuation attempts.
            continue_prompt: User prompt for continuation requests.
            **params: Additional parameters to pass to continue requests.

        Returns:
            Complete result (merged if multiple continues were needed).

        Raises:
            ValueError: If history is not provided or result.finish_reason != "length".

        Examples:
            >>> history = ChatHistory()
            >>> result = chat("Long story", history=history, max_tokens=50)
            >>> # Automatically continue if truncated
            >>> full_result = chat.continue_if_needed(result, history=history)
            >>> # If result.finish_reason != "length", returns result unchanged
        """
        if result.finish_reason != "length":
            return result

        from lexilux.chat.continue_ import ChatContinue

        return ChatContinue.continue_request(
            self,
            result,
            history=history,
            max_continues=max_continues,
            continue_prompt=continue_prompt,
            **params,
        )

    def continue_if_needed_stream(
        self,
        result: ChatResult,
        *,
        history: ChatHistory,
        max_continues: int = 3,
        continue_prompt: str = "continue",
        **params: Any,
    ) -> StreamingIterator:
        """
        Continue generation with streaming if result is truncated (finish_reason == "length").

        This is the streaming version of `continue_if_needed()`. If result.finish_reason != "length",
        returns a StreamingIterator that immediately yields a done chunk with the result.

        Args:
            result: Previous result to check and potentially continue.
            history: ChatHistory instance (required for continuation).
            max_continues: Maximum number of continuation attempts.
            continue_prompt: User prompt for continuation requests.
            **params: Additional parameters to pass to continue requests.

        Returns:
            StreamingIterator: Iterator that yields ChatStreamChunk objects.
                    If result is not truncated, yields a single done chunk.
                    If truncated, yields chunks from all continue requests.
                    Access accumulated result via iterator.result.

        Raises:
            ValueError: If history is not provided.

        Examples:
            >>> history = ChatHistory()
            >>> result = chat("Long story", history=history, max_tokens=50)
            >>> # Stream continue if truncated
            >>> iterator = chat.continue_if_needed_stream(result, history=history)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="", flush=True)
            >>> full_result = iterator.result.to_chat_result()
        """
        from lexilux.chat.continue_ import ChatContinue

        if result.finish_reason != "length":
            # Not truncated, return iterator with single done chunk
            from lexilux.chat.models import ChatStreamChunk
            from lexilux.chat.streaming import StreamingIterator

            def _single_chunk_gen():
                yield ChatStreamChunk(
                    delta="",
                    done=True,
                    usage=result.usage,
                    finish_reason=result.finish_reason,
                )

            iterator = StreamingIterator(_single_chunk_gen())
            # Set result to the original result
            iterator._result._text = result.text
            iterator._result._finish_reason = result.finish_reason
            iterator._result._usage = result.usage
            iterator._result._done = True
            return iterator

        return ChatContinue.continue_request_stream(
            self,
            result,
            history=history,
            max_continues=max_continues,
            continue_prompt=continue_prompt,
            **params,
        )

    def complete(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory,
        max_continues: int = 3,
        ensure_complete: bool = True,
        continue_prompt: str = "continue",
        **params: Any,
    ) -> ChatResult:
        """
        Ensure complete response, automatically handling truncation.

        This is the recommended method for scenarios requiring complete responses
        (e.g., JSON extraction). Automatically handles truncation by continuing
        generation until complete or max_continues reached.

        Args:
            messages: Input messages.
            history: ChatHistory instance (required for automatic continuation).
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues. If False, returns partial result.
            continue_prompt: User prompt for continuation requests.
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            Complete ChatResult (may be result of multiple continues merged).

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.
            ValueError: If history is not provided.

        Examples:
            Ensure complete response (recommended):
            >>> history = ChatHistory()
            >>> result = chat.complete("Write a long JSON", history=history, max_tokens=100)
            >>> # Automatically handles truncation, returns complete result
            >>> json_data = json.loads(result.text)

            Allow partial result:
            >>> history = ChatHistory()
            >>> result = chat.complete(
            ...     "Long story",
            ...     history=history,
            ...     max_tokens=50,
            ...     ensure_complete=False
            ... )
            >>> if result.finish_reason == "length":
            ...     print("Warning: Response was truncated")
        """
        from lexilux.chat.continue_ import ChatContinue
        from lexilux.chat.exceptions import ChatIncompleteResponseError

        result = self(messages, history=history, **params)

        if result.finish_reason == "length":
            try:
                result = ChatContinue.continue_request(
                    self,
                    result,
                    history=history,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
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

        if ensure_complete and result.finish_reason == "length":
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
        history: ChatHistory,
        max_continues: int = 3,
        ensure_complete: bool = True,
        continue_prompt: str = "continue",
        **params: Any,
    ) -> StreamingIterator:
        """
        Ensure complete response with streaming output, automatically handling truncation.

        This is the streaming version of `complete()`. It returns a StreamingIterator
        that yields chunks from the initial request and all continue requests if needed.

        Args:
            messages: Input messages.
            history: ChatHistory instance (required for automatic continuation).
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues. If False, returns partial result.
            continue_prompt: User prompt for continuation requests.
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            StreamingIterator: Iterator that yields ChatStreamChunk objects from
                initial request and all continue requests. Access accumulated result
                via iterator.result.

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.
            ValueError: If history is not provided.

        Examples:
            Ensure complete response with streaming:
            >>> history = ChatHistory()
            >>> iterator = chat.complete_stream("Write a long JSON", history=history, max_tokens=100)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="", flush=True)
            >>> result = iterator.result.to_chat_result()
            >>> json_data = json.loads(result.text)  # Guaranteed complete

            Allow partial result:
            >>> history = ChatHistory()
            >>> iterator = chat.complete_stream(
            ...     "Long story",
            ...     history=history,
            ...     max_tokens=50,
            ...     ensure_complete=False
            ... )
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="", flush=True)
            >>> result = iterator.result.to_chat_result()
            >>> if result.finish_reason == "length":
            ...     print("Warning: Response was truncated")
        """
        from lexilux.chat.continue_ import ChatContinue
        from lexilux.chat.exceptions import ChatIncompleteResponseError

        # Create generator that yields initial chunks and handles continues
        def _complete_stream_generator() -> Iterator[ChatStreamChunk]:
            """Generator that yields chunks from initial request and continues."""
            # Start with streaming request
            initial_iterator = self.stream(messages, history=history, **params)

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
                        history=history,
                        max_continues=max_continues,
                        continue_prompt=continue_prompt,
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
                history: ChatHistory,
                max_continues: int,
                ensure_complete: bool,
            ):
                super().__init__(chunk_gen)
                self._history = history
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
            history,
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
