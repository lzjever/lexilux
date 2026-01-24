"""
Continue functionality for chat completions.

Provides Conversation class for handling continuation requests when generation
is stopped due to max_tokens limit.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.chat.streaming import (
    AsyncStreamingIterator,
    StreamingIterator,
    StreamingResult,
)
from lexilux.usage import Usage

if TYPE_CHECKING:
    from lexilux.chat.client import Chat

logger = logging.getLogger(__name__)


class Conversation:
    """
    Continue functionality handler (user is responsible for determining if continue is needed).

    This class provides utilities for continuing generation when finish_reason == "length".
    The user must check finish_reason and decide when to continue.
    """

    @staticmethod
    def needs_continue(result: ChatResult) -> bool:
        """
        Check if a result needs continuation.

        Args:
            result: ChatResult to check.

        Returns:
            True if result.finish_reason == "length", False otherwise.

        Examples:
            >>> result = chat("Write a story", max_tokens=50)
            >>> if Conversation.needs_continue(result):
            ...     full_result = Conversation.continue_request(chat, result, history=history)
        """
        return result.finish_reason == "length"

    @staticmethod
    def _filter_empty_results(results: list[ChatResult]) -> None:
        results[:] = [
            r for r in results if not (r.text == "" and r.finish_reason is None)
        ]

    @staticmethod
    def _merged_streaming_result(
        initial_result: ChatResult,
        all_results: list[ChatResult],
    ) -> StreamingResult:
        merged_result = (
            Conversation.merge_results(*all_results)
            if len(all_results) > 1
            else initial_result
        )
        streaming_result = StreamingResult()
        streaming_result._text = merged_result.text
        streaming_result._finish_reason = merged_result.finish_reason
        streaming_result._usage = merged_result.usage
        streaming_result._done = True
        return streaming_result

    @staticmethod
    def _get_continue_prompt(
        continue_prompt: str | Callable,
        continue_count: int,
        max_continues: int,
        current_text: str,
        original_prompt: str | None = None,
    ) -> str:
        """
        Get continue prompt (supports string or callable).

        Args:
            continue_prompt: String or callable that generates the prompt.
            continue_count: Current continue count (1-indexed).
            max_continues: Maximum number of continues.
            current_text: Current accumulated text.
            original_prompt: Original user prompt (if available).

        Returns:
            Continue prompt string.
        """
        if callable(continue_prompt):
            return continue_prompt(
                continue_count, max_continues, current_text, original_prompt or ""
            )
        return continue_prompt

    @staticmethod
    def _call_progress_callback(
        on_progress: Callable | None,
        continue_count: int,
        max_continues: int,
        current_result: ChatResult,
        all_results: list[ChatResult],
    ) -> None:
        """
        Call progress callback if provided.

        Args:
            on_progress: Progress callback function.
            continue_count: Current continue count.
            max_continues: Maximum number of continues.
            current_result: Current result.
            all_results: All results so far.
        """
        if on_progress:
            try:
                on_progress(continue_count, max_continues, current_result, all_results)
            # Catch any exception from user-provided callback to avoid breaking main flow
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    @staticmethod
    def _apply_continue_delay(
        continue_delay: float | tuple[float, float],
        continue_count: int,
    ) -> None:
        """
        Apply continue delay if needed.

        Args:
            continue_delay: Fixed delay (seconds) or tuple (min, max) for random delay.
            continue_count: Current continue count (delay only applied if count > 1).
        """
        delay = Conversation._get_delay_seconds(continue_delay, continue_count)
        if delay is not None:
            time.sleep(delay)

    @staticmethod
    def _get_delay_seconds(
        continue_delay: float | tuple[float, float],
        continue_count: int,
    ) -> float | None:
        if continue_count <= 1:
            return None
        delay = (
            random.uniform(continue_delay[0], continue_delay[1])
            if isinstance(continue_delay, tuple)
            else continue_delay
        )
        return delay if delay > 0 else None

    @staticmethod
    def _handle_continue_error(
        error: Exception,
        partial_result: ChatResult,
        all_results: list[ChatResult],
        on_error: str,
        on_error_callback: Callable | None,
    ) -> ChatResult:
        """
        Handle continue error based on strategy.

        Args:
            error: Exception that occurred.
            partial_result: Partial result before error.
            all_results: All results collected so far.
            on_error: Error strategy ("raise" or "return_partial").
            on_error_callback: Optional error callback function.

        Returns:
            ChatResult if returning partial, otherwise raises exception.

        Raises:
            Exception: If strategy is "raise" or callback returns "raise".
        """
        action = on_error
        callback_result: ChatResult | None = None

        if on_error_callback:
            try:
                response = on_error_callback(error, partial_result)
                if isinstance(response, dict):
                    action = response.get("action", action)
                    result = response.get("result")
                    if isinstance(result, ChatResult):
                        callback_result = result
            except Exception as callback_error:
                logger.warning(f"Error callback failed: {callback_error}")

        if callback_result is not None:
            return callback_result

        if action == "return_partial":
            return (
                Conversation.merge_results(*all_results)
                if len(all_results) > 1
                else partial_result
            )

        if action == "retry":
            raise NotImplementedError("Retry action not implemented") from error

        raise error

    @staticmethod
    def _prepare_continue_loop(
        last_result: ChatResult,
        messages: list[dict[str, Any]],
        original_prompt: str | None,
    ) -> tuple[
        list[dict[str, Any]], list[ChatResult], ChatResult, int, str, str | None
    ]:
        """
        Prepare state for the continuation loop.

        Args:
            last_result: The result that needs continuation.
            messages: Current messages list (modified in place for continuation).
            original_prompt: Original user prompt for dynamic continue prompts.

        Returns:
            Tuple of (messages, all_results, current_result, continue_count,
                     accumulated_text, original_prompt).
        """
        if last_result.finish_reason != "length":
            raise ValueError(
                f"continue_request requires finish_reason='length', got '{last_result.finish_reason}'"
            )
        if not messages:
            raise ValueError("Messages list is required for continuation requests.")

        all_results = [last_result]
        current_result = last_result
        continue_count = 0
        accumulated_text = last_result.text

        if original_prompt is None:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    original_prompt = msg.get("content", "")
                    break

        return (
            messages,
            all_results,
            current_result,
            continue_count,
            accumulated_text,
            original_prompt,
        )

    @staticmethod
    def _process_continue_step(
        continue_result: ChatResult,
        all_results: list[ChatResult],
        messages: list[dict[str, Any]],
    ) -> tuple[ChatResult, str]:
        """Process the result of a single continuation step."""
        all_results.append(continue_result)
        current_result = continue_result
        accumulated_text = "".join(r.text for r in all_results)
        # Append assistant response to messages list
        messages.append({"role": "assistant", "content": continue_result.text})
        return current_result, accumulated_text

    @staticmethod
    def _finalize_continue_request(
        current_result: ChatResult, all_results: list[ChatResult], auto_merge: bool
    ) -> ChatResult | list[ChatResult]:
        """Finalize the continuation request, merging results if needed."""
        if current_result.finish_reason == "length" and not auto_merge:
            return all_results

        if auto_merge:
            return (
                all_results[0]
                if len(all_results) == 1
                else Conversation.merge_results(*all_results)
            )
        return all_results

    @staticmethod
    @overload
    def continue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        messages: list[dict[str, Any]],
        add_continue_prompt: bool = True,
        continue_prompt: str | Callable = "continue",
        max_continues: int = 1,
        auto_merge: Literal[True] = True,
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        original_prompt: str | None = None,
        **params: Any,
    ) -> ChatResult: ...

    @staticmethod
    @overload
    def continue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        messages: list[dict[str, Any]],
        add_continue_prompt: bool = True,
        continue_prompt: str | Callable = "continue",
        max_continues: int = 1,
        auto_merge: Literal[False],
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        original_prompt: str | None = None,
        **params: Any,
    ) -> list[ChatResult]: ...

    @staticmethod
    def continue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        messages: list[dict[str, Any]],
        add_continue_prompt: bool = True,
        continue_prompt: str | Callable = "continue",
        max_continues: int = 1,
        auto_merge: bool = True,
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        original_prompt: str | None = None,
        **params: Any,
    ) -> ChatResult | list[ChatResult]:
        """
        Continue generation request (enhanced version with customization support).

        Args:
            chat: Chat client instance.
            last_result: The result that needs continuation.
            messages: Messages list (modified in place for continuation tracking).
            add_continue_prompt: Whether to add continue prompt.
            continue_prompt: Continue prompt string or callable.
            max_continues: Maximum continuation attempts.
            auto_merge: Whether to merge results automatically.
            on_progress: Progress callback.
            continue_delay: Delay between requests.
            on_error: Error handling strategy.
            on_error_callback: Error callback.
            original_prompt: Original user prompt.
            **params: Additional parameters for chat calls.
        """
        (
            working_messages,
            all_results,
            current_result,
            continue_count,
            accumulated_text,
            original_prompt,
        ) = Conversation._prepare_continue_loop(last_result, messages, original_prompt)

        while (
            current_result.finish_reason == "length" and continue_count < max_continues
        ):
            continue_count += 1
            Conversation._apply_continue_delay(continue_delay, continue_count)
            Conversation._call_progress_callback(
                on_progress, continue_count, max_continues, current_result, all_results
            )

            try:
                prompt = Conversation._get_continue_prompt(
                    continue_prompt,
                    continue_count,
                    max_continues,
                    accumulated_text,
                    original_prompt or "",
                )
                if add_continue_prompt:
                    working_messages.append({"role": "user", "content": prompt})

                continue_result = chat(working_messages, **params)

                current_result, accumulated_text = Conversation._process_continue_step(
                    continue_result, all_results, working_messages
                )
            except Exception as e:
                return Conversation._handle_continue_error(
                    e, current_result, all_results, on_error, on_error_callback
                )

        return Conversation._finalize_continue_request(
            current_result, all_results, auto_merge
        )

    @staticmethod
    def merge_results(*results: ChatResult) -> ChatResult:
        """
        Merge multiple ChatResult instances into a single result.

        Args:
            *results: Multiple ChatResult instances to merge in order.

        Returns:
            Merged ChatResult with combined text and usage.

        Examples:
            >>> result1 = chat("Write a story", max_tokens=50)
            >>> result2 = chat.continue_request(...)
            >>> full_result = Conversation.merge_results(result1, result2)
        """
        if not results:
            raise ValueError("At least one result is required")

        if len(results) == 1:
            return results[0]

        # Merge text
        merged_text = "".join(r.text for r in results)

        # Merge usage
        total_input_tokens = sum(
            r.usage.input_tokens or 0
            for r in results
            if r.usage.input_tokens is not None
        )
        total_output_tokens = sum(
            r.usage.output_tokens or 0
            for r in results
            if r.usage.output_tokens is not None
        )
        total_tokens = sum(
            r.usage.total_tokens or 0
            for r in results
            if r.usage.total_tokens is not None
        )

        # Use last result's finish_reason (most recent)
        finish_reason = results[-1].finish_reason

        # Merge raw data (combine details)
        merged_raw = {}
        for r in results:
            if r.raw:
                merged_raw.update(r.raw)

        merged_usage = Usage(
            input_tokens=total_input_tokens if total_input_tokens > 0 else None,
            output_tokens=total_output_tokens if total_output_tokens > 0 else None,
            total_tokens=total_tokens if total_tokens > 0 else None,
            details=merged_raw,
        )

        return ChatResult(
            text=merged_text,
            usage=merged_usage,
            finish_reason=finish_reason,
            raw=merged_raw,
        )

    @staticmethod
    def continue_request_stream(
        chat: Chat,
        last_result: ChatResult,
        *,
        messages: list[dict[str, Any]],
        add_continue_prompt: bool = True,
        continue_prompt: str | Callable = "continue",
        max_continues: int = 1,
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        original_prompt: str | None = None,
        **params: Any,
    ) -> StreamingIterator:
        """
        Continue generation with streaming output (enhanced version with customization support).

        Args:
            chat: Chat client instance.
            last_result: The result that needs continuation.
            messages: Messages list (modified in place for continuation tracking).
            add_continue_prompt: Whether to add continue prompt.
            continue_prompt: Continue prompt string or callable.
            max_continues: Maximum continuation attempts.
            on_progress: Progress callback.
            continue_delay: Delay between requests.
            on_error: Error handling strategy.
            on_error_callback: Error callback.
            original_prompt: Original user prompt.
            **params: Additional parameters for chat calls.
        """
        (
            working_messages,
            all_results,
            current_result,
            continue_count,
            accumulated_text,
            original_prompt,
        ) = Conversation._prepare_continue_loop(last_result, messages, original_prompt)

        # Reset all_results for the generator to populate
        all_results.clear()
        all_results.append(last_result)

        def _continue_chunk_generator() -> Iterator[ChatStreamChunk]:
            """Generator that yields chunks from all continue requests."""
            nonlocal current_result, continue_count, accumulated_text
            while (
                current_result.finish_reason == "length"
                and continue_count < max_continues
            ):
                continue_count += 1
                Conversation._apply_continue_delay(continue_delay, continue_count)
                Conversation._call_progress_callback(
                    on_progress,
                    continue_count,
                    max_continues,
                    current_result,
                    all_results,
                )

                try:
                    prompt = Conversation._get_continue_prompt(
                        continue_prompt,
                        continue_count,
                        max_continues,
                        accumulated_text,
                        original_prompt or "",
                    )
                    if add_continue_prompt:
                        working_messages.append({"role": "user", "content": prompt})

                    continue_iterator = chat.stream(working_messages, **params)
                    yield from continue_iterator

                    continue_result = continue_iterator.result.to_chat_result()
                    current_result, accumulated_text = (
                        Conversation._process_continue_step(
                            continue_result, all_results, working_messages
                        )
                    )
                except Exception as e:
                    Conversation._handle_continue_error(
                        e, current_result, all_results, on_error, on_error_callback
                    )
                    break

        return _MergedContinueIterator(
            _continue_chunk_generator(), last_result, all_results
        )

    # =========================================================================
    # Async Methods
    # =========================================================================

    @staticmethod
    async def _apply_continue_delay_async(
        continue_delay: float | tuple[float, float],
        continue_count: int,
    ) -> None:
        """
        Apply continue delay asynchronously.
        """
        delay = Conversation._get_delay_seconds(continue_delay, continue_count)
        if delay is not None:
            await asyncio.sleep(delay)

    @staticmethod
    async def acontinue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        messages: list[dict[str, Any]],
        add_continue_prompt: bool = True,
        continue_prompt: str | Callable = "continue",
        max_continues: int = 1,
        auto_merge: bool = True,
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        original_prompt: str | None = None,
        **params: Any,
    ) -> ChatResult | list[ChatResult]:
        """
        Async version of continue_request.

        Args:
            chat: Chat client instance.
            last_result: The result that needs continuation.
            messages: Messages list (modified in place for continuation tracking).
            add_continue_prompt: Whether to add continue prompt.
            continue_prompt: Continue prompt string or callable.
            max_continues: Maximum continuation attempts.
            auto_merge: Whether to merge results automatically.
            on_progress: Progress callback.
            continue_delay: Delay between requests.
            on_error: Error handling strategy.
            on_error_callback: Error callback.
            original_prompt: Original user prompt.
            **params: Additional parameters for chat calls.
        """
        (
            working_messages,
            all_results,
            current_result,
            continue_count,
            accumulated_text,
            original_prompt,
        ) = Conversation._prepare_continue_loop(last_result, messages, original_prompt)

        while (
            current_result.finish_reason == "length" and continue_count < max_continues
        ):
            continue_count += 1
            await Conversation._apply_continue_delay_async(
                continue_delay, continue_count
            )
            Conversation._call_progress_callback(
                on_progress, continue_count, max_continues, current_result, all_results
            )

            try:
                prompt = Conversation._get_continue_prompt(
                    continue_prompt,
                    continue_count,
                    max_continues,
                    accumulated_text,
                    original_prompt or "",
                )
                if add_continue_prompt:
                    working_messages.append({"role": "user", "content": prompt})

                continue_result = await chat.acall(working_messages, **params)

                current_result, accumulated_text = Conversation._process_continue_step(
                    continue_result, all_results, working_messages
                )
            except Exception as e:
                return Conversation._handle_continue_error(
                    e, current_result, all_results, on_error, on_error_callback
                )

        return Conversation._finalize_continue_request(
            current_result, all_results, auto_merge
        )

    @staticmethod
    async def acontinue_request_stream(
        chat: Chat,
        last_result: ChatResult,
        *,
        messages: list[dict[str, Any]],
        add_continue_prompt: bool = True,
        continue_prompt: str | Callable = "continue",
        max_continues: int = 1,
        on_progress: Callable | None = None,
        continue_delay: float | tuple[float, float] = 0.0,
        on_error: str = "raise",
        on_error_callback: Callable | None = None,
        original_prompt: str | None = None,
        **params: Any,
    ) -> AsyncStreamingIterator:
        """
        Async version of continue_request_stream.

        Args:
            chat: Chat client instance.
            last_result: The result that needs continuation.
            messages: Messages list (modified in place for continuation tracking).
            add_continue_prompt: Whether to add continue prompt.
            continue_prompt: Continue prompt string or callable.
            max_continues: Maximum continuation attempts.
            on_progress: Progress callback.
            continue_delay: Delay between requests.
            on_error: Error handling strategy.
            on_error_callback: Error callback.
            original_prompt: Original user prompt.
            **params: Additional parameters for chat calls.
        """
        (
            working_messages,
            all_results,
            current_result,
            continue_count,
            accumulated_text,
            original_prompt,
        ) = Conversation._prepare_continue_loop(last_result, messages, original_prompt)

        all_results.clear()
        all_results.append(last_result)

        async def _async_continue_chunk_generator() -> AsyncIterator[ChatStreamChunk]:
            """Async generator that yields chunks from all continue requests."""
            nonlocal current_result, continue_count, accumulated_text
            while (
                current_result.finish_reason == "length"
                and continue_count < max_continues
            ):
                continue_count += 1
                await Conversation._apply_continue_delay_async(
                    continue_delay, continue_count
                )
                Conversation._call_progress_callback(
                    on_progress,
                    continue_count,
                    max_continues,
                    current_result,
                    all_results,
                )

                try:
                    prompt = Conversation._get_continue_prompt(
                        continue_prompt,
                        continue_count,
                        max_continues,
                        accumulated_text,
                        original_prompt or "",
                    )
                    if add_continue_prompt:
                        working_messages.append({"role": "user", "content": prompt})

                    continue_iterator = await chat.astream(working_messages, **params)
                    async for chunk in continue_iterator:
                        yield chunk

                    continue_result = continue_iterator.result.to_chat_result()
                    current_result, accumulated_text = (
                        Conversation._process_continue_step(
                            continue_result, all_results, working_messages
                        )
                    )
                except Exception as e:
                    Conversation._handle_continue_error(
                        e, current_result, all_results, on_error, on_error_callback
                    )
                    break

        return _AsyncMergedContinueIterator(
            _async_continue_chunk_generator(), last_result, all_results
        )


class _BaseMergedContinueIterator:
    """Base class for MergedContinueIterator and AsyncMergedContinueIterator."""

    def __init__(
        self,
        initial_result: ChatResult,
        all_results_ref: list[ChatResult],
    ):
        self._initial_result = initial_result
        self._all_results_ref = all_results_ref
        self._merged_result: StreamingResult | None = None

    @property
    def result(self) -> StreamingResult:
        """Get merged result from all continues."""
        if self._merged_result is None:
            self._merged_result = Conversation._merged_streaming_result(
                self._initial_result,
                self._all_results_ref,
            )
        return self._merged_result


class _MergedContinueIterator(StreamingIterator, _BaseMergedContinueIterator):
    """Iterator that merges results from all continue requests."""

    def __init__(
        self,
        chunk_gen: Iterator[ChatStreamChunk],
        initial_result: ChatResult,
        all_results_ref: list[ChatResult],
    ):
        StreamingIterator.__init__(self, chunk_gen)
        _BaseMergedContinueIterator.__init__(self, initial_result, all_results_ref)

    def __iter__(self) -> Iterator[ChatStreamChunk]:
        """Iterate chunks."""
        for chunk in self._iterator:
            self._result.update(chunk)
            yield chunk
        if self._all_results_ref:
            Conversation._filter_empty_results(self._all_results_ref)


class _AsyncMergedContinueIterator(AsyncStreamingIterator, _BaseMergedContinueIterator):
    """Async iterator that merges results from all continue requests."""

    def __init__(
        self,
        chunk_gen: AsyncIterator[ChatStreamChunk],
        initial_result: ChatResult,
        all_results_ref: list[ChatResult],
    ):
        AsyncStreamingIterator.__init__(self, chunk_gen)
        _BaseMergedContinueIterator.__init__(self, initial_result, all_results_ref)

    async def __anext__(self) -> ChatStreamChunk:
        try:
            chunk = await self._iterator.__anext__()
            self._result.update(chunk)
            return chunk
        except StopAsyncIteration:
            if self._all_results_ref:
                Conversation._filter_empty_results(self._all_results_ref)
            raise
