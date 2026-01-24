"""
Completion helpers for the Chat client.

This module contains the "ensure complete" APIs (continue-on-length) and keeps
the main client focused on a single request/stream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TYPE_CHECKING
from lexilux.chat.conversation import Conversation
from lexilux.chat.exceptions import ChatIncompleteResponseError
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike
from lexilux.chat.streaming import AsyncStreamingIterator, StreamingIterator

if TYPE_CHECKING:
    from lexilux.chat.client import Chat


def _get_working_history(history: ChatHistory | None) -> ChatHistory:
    return history.clone() if history is not None else ChatHistory()


def _get_original_prompt(messages: MessagesLike) -> str:
    return messages if isinstance(messages, str) else str(messages)


def complete(
    chat: Chat,
    messages: MessagesLike,
    *,
    history: ChatHistory | None,
    max_continues: int,
    ensure_complete: bool,
    continue_prompt: str | Callable,
    on_progress: Callable | None,
    continue_delay: float | tuple[float, float],
    on_error: str,
    on_error_callback: Callable | None,
    params: dict[str, Any],
) -> ChatResult:
    working_history = _get_working_history(history)
    original_prompt = _get_original_prompt(messages)

    result = chat(messages, history=working_history, **params)
    if result.finish_reason == "length":
        try:
            result = Conversation.continue_request(
                chat,
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


class CompleteStreamingIterator(StreamingIterator):
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


def complete_stream(
    chat: Chat,
    messages: MessagesLike,
    *,
    history: ChatHistory | None,
    max_continues: int,
    ensure_complete: bool,
    continue_prompt: str | Callable,
    on_progress: Callable | None,
    continue_delay: float | tuple[float, float],
    on_error: str,
    on_error_callback: Callable | None,
    params: dict[str, Any],
) -> StreamingIterator:
    working_history = _get_working_history(history)
    original_prompt = _get_original_prompt(messages)

    def _complete_stream_generator() -> Iterator[ChatStreamChunk]:
        initial_iterator = chat.stream(messages, history=working_history, **params)
        yield from initial_iterator

        initial_result = initial_iterator.result.to_chat_result()
        if initial_result.finish_reason != "length":
            return

        try:
            continue_iterator = Conversation.continue_request_stream(
                chat,
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

    return CompleteStreamingIterator(
        _complete_stream_generator(),
        max_continues=max_continues,
        ensure_complete=ensure_complete,
    )


async def acomplete(
    chat: Chat,
    messages: MessagesLike,
    *,
    history: ChatHistory | None,
    max_continues: int,
    ensure_complete: bool,
    continue_prompt: str | Callable,
    on_progress: Callable | None,
    continue_delay: float | tuple[float, float],
    on_error: str,
    on_error_callback: Callable | None,
    params: dict[str, Any],
) -> ChatResult:
    working_history = _get_working_history(history)
    original_prompt = _get_original_prompt(messages)

    result = await chat.acall(messages, history=working_history, **params)
    if result.finish_reason == "length":
        try:
            result = await Conversation.acontinue_request(
                chat,
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


class AsyncCompleteStreamingIterator(AsyncStreamingIterator):
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


async def acomplete_stream(
    chat: Chat,
    messages: MessagesLike,
    *,
    history: ChatHistory | None,
    max_continues: int,
    ensure_complete: bool,
    continue_prompt: str | Callable,
    on_progress: Callable | None,
    continue_delay: float | tuple[float, float],
    on_error: str,
    on_error_callback: Callable | None,
    params: dict[str, Any],
) -> AsyncStreamingIterator:
    working_history = _get_working_history(history)
    original_prompt = _get_original_prompt(messages)

    async def _async_complete_stream_generator() -> AsyncIterator[ChatStreamChunk]:
        initial_iterator = await chat.astream(
            messages, history=working_history, **params
        )
        async for chunk in initial_iterator:
            yield chunk

        initial_result = initial_iterator.result.to_chat_result()
        if initial_result.finish_reason != "length":
            return

        try:
            continue_iterator = await Conversation.acontinue_request_stream(
                chat,
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

    return AsyncCompleteStreamingIterator(
        _async_complete_stream_generator(),
        max_continues=max_continues,
        ensure_complete=ensure_complete,
    )
