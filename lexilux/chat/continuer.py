"""
Conversation continuation functionality.

Provides ConversationContinuer class for handling conversation continuation
logic with history management.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any

from lexilux.chat.conversation import Conversation
from lexilux.chat.exceptions import ChatIncompleteResponseError
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike
from lexilux.chat.streaming import (
    AsyncStreamingIterator,
    StreamingIterator,
)
from lexilux.exceptions import LexiluxError

if TYPE_CHECKING:
    from lexilux.chat.client import Chat

logger = logging.getLogger(__name__)


def _get_original_prompt(messages: MessagesLike) -> str:
    """Extract original prompt from messages."""
    return messages if isinstance(messages, str) else str(messages)


class ConversationContinuer:
    """
    Handles conversation continuation logic.

    Manages working history and state for complete() methods.
    Delegates to the Conversation class for actual continuation requests.
    """

    def __init__(self, client: "Chat"):
        """
        Initialize conversation continuer.

        Args:
            client: Parent Chat client instance
        """
        self._client = client

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
        Complete a conversation with automatic continuation handling.

        Ensures a complete response by automatically handling truncation.
        Manages history updates throughout the continuation process.

        Args:
            messages: Input messages.
            history: Optional ChatHistory instance. If None, creates a new one internally.
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues.
            continue_prompt: User prompt for continuation requests.
            on_progress: Optional progress callback function.
            continue_delay: Delay between continue requests (seconds).
            on_error: Error handling strategy: "raise" or "return_partial".
            on_error_callback: Optional error callback function.
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            Complete ChatResult (never truncated, unless max_continues exceeded).

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.
        """
        from lexilux.chat.utils import normalize_messages

        original_prompt = _get_original_prompt(messages)

        # Build messages list (read-only from history, no cloning)
        working_messages: list[dict[str, Any]] = []
        if history is not None:
            working_messages.extend(history.get_messages(include_system=True))

        # Add user message(s) from input
        for msg in normalize_messages(messages):
            working_messages.append(msg)

        # First API call
        result = self._client(working_messages, **params)

        # Add assistant response for potential continuation
        working_messages.append({"role": "assistant", "content": result.text})

        if result.finish_reason == "length":
            try:
                result = Conversation.continue_request(
                    self._client,
                    result,
                    messages=working_messages,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except LexiluxError as e:
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
        Complete a conversation with streaming and automatic continuation handling.

        Ensures a complete response by automatically handling truncation during streaming.
        Manages history updates throughout the continuation process.

        Args:
            messages: Input messages.
            history: Optional ChatHistory instance. If None, creates a new one internally.
            max_continues: Maximum number of continuation attempts.
            ensure_complete: If True, raises ChatIncompleteResponseError if result is still
                truncated after max_continues.
            continue_prompt: User prompt for continuation requests.
            on_progress: Optional progress callback function.
            continue_delay: Delay between continue requests (seconds).
            on_error: Error handling strategy: "raise" or "return_partial".
            on_error_callback: Optional error callback function.
            **params: Additional parameters to pass to chat and continue requests.

        Returns:
            StreamingIterator that yields chunks from initial request and all continues.

        Raises:
            ChatIncompleteResponseError: If ensure_complete=True and result is still truncated
                after max_continues.
        """
        from lexilux.chat.utils import normalize_messages

        original_prompt = _get_original_prompt(messages)

        # Build messages list (read-only from history, no cloning)
        working_messages: list[dict[str, Any]] = []
        if history is not None:
            working_messages.extend(history.get_messages(include_system=True))

        # Add user message(s) from input
        for msg in normalize_messages(messages):
            working_messages.append(msg)

        def _complete_stream_generator() -> Iterator[ChatStreamChunk]:
            # First stream
            initial_iterator = self._client.stream(working_messages, **params)
            yield from initial_iterator

            initial_result = initial_iterator.result.to_chat_result()

            # Add assistant response for potential continuation
            working_messages.append(
                {"role": "assistant", "content": initial_result.text}
            )

            if initial_result.finish_reason != "length":
                return

            try:
                continue_iterator = Conversation.continue_request_stream(
                    self._client,
                    initial_result,
                    messages=working_messages,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except LexiluxError as e:
                if ensure_complete:
                    raise ChatIncompleteResponseError(
                        f"Failed to get complete response after {max_continues} continues: {e}",
                        final_result=initial_result,
                        continue_count=0,
                        max_continues=max_continues,
                    ) from e
                raise

            yield from continue_iterator

        from lexilux.chat.streaming import CompleteStreamingIterator

        return CompleteStreamingIterator(
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

        Complete a conversation asynchronously with automatic continuation handling.
        """
        from lexilux.chat.utils import normalize_messages

        original_prompt = _get_original_prompt(messages)

        # Build messages list (read-only from history, no cloning)
        working_messages: list[dict[str, Any]] = []
        if history is not None:
            working_messages.extend(history.get_messages(include_system=True))

        # Add user message(s) from input
        for msg in normalize_messages(messages):
            working_messages.append(msg)

        # First API call
        result = await self._client.acall(working_messages, **params)

        # Add assistant response for potential continuation
        working_messages.append({"role": "assistant", "content": result.text})

        if result.finish_reason == "length":
            try:
                result = await Conversation.acontinue_request(
                    self._client,
                    result,
                    messages=working_messages,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except LexiluxError as e:
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

        Complete a conversation asynchronously with streaming and automatic continuation handling.
        """
        from lexilux.chat.utils import normalize_messages

        original_prompt = _get_original_prompt(messages)

        # Build messages list (read-only from history, no cloning)
        working_messages: list[dict[str, Any]] = []
        if history is not None:
            working_messages.extend(history.get_messages(include_system=True))

        # Add user message(s) from input
        for msg in normalize_messages(messages):
            working_messages.append(msg)

        async def _async_complete_stream_generator() -> AsyncIterator[ChatStreamChunk]:
            # First stream
            initial_iterator = await self._client.astream(working_messages, **params)
            async for chunk in initial_iterator:
                yield chunk

            initial_result = initial_iterator.result.to_chat_result()

            # Add assistant response for potential continuation
            working_messages.append(
                {"role": "assistant", "content": initial_result.text}
            )

            if initial_result.finish_reason != "length":
                return

            try:
                continue_iterator = await Conversation.acontinue_request_stream(
                    self._client,
                    initial_result,
                    messages=working_messages,
                    max_continues=max_continues,
                    continue_prompt=continue_prompt,
                    on_progress=on_progress,
                    continue_delay=continue_delay,
                    on_error=on_error,
                    on_error_callback=on_error_callback,
                    original_prompt=original_prompt,
                    **params,
                )
            except LexiluxError as e:
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

        from lexilux.chat.streaming import AsyncCompleteStreamingIterator

        return AsyncCompleteStreamingIterator(
            _async_complete_stream_generator(),
            max_continues=max_continues,
            ensure_complete=ensure_complete,
        )
