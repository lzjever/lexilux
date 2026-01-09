"""
Continue functionality for chat completions.

Provides ChatContinue class for handling continuation requests when generation
is stopped due to max_tokens limit.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, overload

from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.chat.streaming import StreamingIterator, StreamingResult
from lexilux.usage import Usage

if TYPE_CHECKING:
    from lexilux.chat.client import Chat

logger = logging.getLogger(__name__)


class ChatContinue:
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
            >>> if ChatContinue.needs_continue(result):
            ...     full_result = ChatContinue.continue_request(chat, result, history=history)
        """
        return result.finish_reason == "length"

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
            except Exception as e:
                # Callback failure should not affect main flow
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
        if continue_count <= 1:
            return  # No delay for first continue

        if isinstance(continue_delay, tuple):
            # Random delay range
            delay = random.uniform(continue_delay[0], continue_delay[1])
        else:
            # Fixed delay
            delay = continue_delay

        if delay > 0:
            time.sleep(delay)

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
        if on_error_callback:
            try:
                response = on_error_callback(error, partial_result)
                if isinstance(response, dict):
                    action = response.get("action", "raise")
                    if action == "return_partial":
                        if len(all_results) > 1:
                            return ChatContinue.merge_results(*all_results)
                        return partial_result
                    elif action == "retry":
                        # Retry not implemented yet
                        raise NotImplementedError("Retry action not implemented")
                    # else: "raise" - fall through
            except Exception as callback_error:
                logger.warning(f"Error callback failed: {callback_error}")
                # Fall through to default behavior

        if on_error == "return_partial":
            if len(all_results) > 1:
                return ChatContinue.merge_results(*all_results)
            return partial_result
        else:  # "raise"
            raise

    @staticmethod
    @overload
    def continue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        history: ChatHistory | None = None,
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
        history: ChatHistory | None = None,
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
        history: ChatHistory | None = None,
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

        Automatically handles continuation when finish_reason == "length", with support
        for multiple continues, automatic merging, and customizable strategies.

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        Args:
            chat: Chat client instance.
            last_result: Last result (must have finish_reason == "length").
            history: Conversation history (required). Must be provided explicitly.
            add_continue_prompt: Whether to add a user continue instruction round.
            continue_prompt: User prompt when add_continue_prompt=True. Can be a string or
                a callable with signature: (count: int, max_count: int, current_text: str, original_prompt: str) -> str
            max_continues: Maximum number of continuation attempts. If result is still
                truncated after max_continues, returns merged result (if auto_merge=True)
                or list of results (if auto_merge=False).
            auto_merge: If True, automatically merge all results into a single ChatResult.
                If False, returns a list of all results [last_result, continue_result1, ...].
            on_progress: Optional progress callback function with signature:
                (count: int, max_count: int, current_result: ChatResult, all_results: list[ChatResult]) -> None
            continue_delay: Delay between continue requests (seconds). Can be a float (fixed delay)
                or tuple (min, max) for random delay. Delay is only applied after the first continue.
            on_error: Error handling strategy: "raise" (default) or "return_partial".
            on_error_callback: Optional error callback function with signature:
                (error: Exception, partial_result: ChatResult) -> dict
                Should return {"action": "raise" | "return_partial" | "retry", "result": ChatResult}
            original_prompt: Original user prompt (used for dynamic continue_prompt generation).
            **params: Additional parameters to pass to chat (temperature, max_tokens, etc.).

        Returns:
            If auto_merge=True: Merged ChatResult with all continuation results combined.
            If auto_merge=False: List of ChatResult instances [last_result, continue_result1, ...].

        Raises:
            ValueError: If last_result.finish_reason != "length" or history is not provided.
            Exception: If on_error="raise" and an error occurs during continuation.

        Examples:
            Basic usage:
            >>> history = ChatHistory()
            >>> result = chat("Write a long story", history=history, max_tokens=50)
            >>> if result.finish_reason == "length":
            ...     full_result = ChatContinue.continue_request(chat, result, history=history)
            ...     print(full_result.text)  # Complete merged text

            With progress tracking:
            >>> def on_progress(count, max_count, current, all_results):
            ...     print(f"继续生成 {count}/{max_count}...")
            >>> full_result = ChatContinue.continue_request(
            ...     chat, result, history=history,
            ...     on_progress=on_progress
            ... )

            With custom prompt and delay:
            >>> def smart_prompt(count, max_count, current_text, original_prompt):
            ...     return f"请继续完成（第 {count}/{max_count} 次）"
            >>> full_result = ChatContinue.continue_request(
            ...     chat, result, history=history,
            ...     continue_prompt=smart_prompt,
            ...     continue_delay=(1.0, 2.0)  # Random delay 1-2 seconds
            ... )
        """
        if last_result.finish_reason != "length":
            raise ValueError(
                f"continue_request requires finish_reason='length', "
                f"got '{last_result.finish_reason}'"
            )

        if history is None:
            raise ValueError(
                "History is required. Provide history explicitly when calling continue_request."
            )

        # Create working history (immutable - clone original)
        working_history = history.clone()

        all_results = [last_result]
        current_result = last_result
        continue_count = 0
        accumulated_text = last_result.text

        # Get original prompt from history if not provided
        if original_prompt is None:
            history_messages = working_history.get_messages()
            for msg in reversed(history_messages):
                if msg.get("role") == "user":
                    original_prompt = msg.get("content", "")
                    break

        while current_result.finish_reason == "length" and continue_count < max_continues:
            continue_count += 1

            # Apply delay if needed (not for first continue)
            ChatContinue._apply_continue_delay(continue_delay, continue_count)

            # Call progress callback
            ChatContinue._call_progress_callback(
                on_progress, continue_count, max_continues, current_result, all_results
            )

            try:
                # Get continue prompt (supports string or callable)
                prompt = ChatContinue._get_continue_prompt(
                    continue_prompt,
                    continue_count,
                    max_continues,
                    accumulated_text,
                    original_prompt or "",
                )

                # Execute single continue request
                if add_continue_prompt:
                    working_history.add_user(prompt)

                continue_result = chat(
                    working_history.get_messages(), history=working_history, **params
                )
                all_results.append(continue_result)
                current_result = continue_result
                accumulated_text += continue_result.text

            except Exception as e:
                # Handle error based on strategy
                try:
                    result = ChatContinue._handle_continue_error(
                        e, current_result, all_results, on_error, on_error_callback
                    )
                    return result if auto_merge else all_results
                except Exception:
                    # Re-raise if strategy is "raise"
                    raise

        # Check if still truncated after max_continues
        if current_result.finish_reason == "length":
            if auto_merge:
                # Return merged result even if truncated
                return ChatContinue.merge_results(*all_results)
            else:
                # Return all results, let user decide
                return all_results

        # Merge results if auto_merge
        if auto_merge:
            if len(all_results) == 1:
                return all_results[0]
            return ChatContinue.merge_results(*all_results)
        else:
            return all_results

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
            >>> full_result = ChatContinue.merge_results(result1, result2)
        """
        if not results:
            raise ValueError("At least one result is required")

        if len(results) == 1:
            return results[0]

        # Merge text
        merged_text = "".join(r.text for r in results)

        # Merge usage
        total_input_tokens = sum(
            r.usage.input_tokens or 0 for r in results if r.usage.input_tokens is not None
        )
        total_output_tokens = sum(
            r.usage.output_tokens or 0 for r in results if r.usage.output_tokens is not None
        )
        total_tokens = sum(
            r.usage.total_tokens or 0 for r in results if r.usage.total_tokens is not None
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
        history: ChatHistory,
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

        This is the streaming version of `continue_request()`. It returns a
        StreamingIterator that yields chunks for all continuation requests.

        **History Immutability**: If history is provided, a clone is created and used internally.
        The original history is never modified.

        Args:
            chat: Chat client instance.
            last_result: Last result (must have finish_reason == "length").
            history: Conversation history (required). Must be provided explicitly.
            add_continue_prompt: Whether to add a user continue instruction round.
            continue_prompt: User prompt when add_continue_prompt=True. Can be a string or
                a callable with signature: (count: int, max_count: int, current_text: str, original_prompt: str) -> str
            max_continues: Maximum number of continuation attempts. If result is still truncated after max_continues, returns merged result.
            on_progress: Optional progress callback function with signature:
                (count: int, max_count: int, current_result: ChatResult, all_results: list[ChatResult]) -> None
            continue_delay: Delay between continue requests (seconds). Can be a float (fixed delay)
                or tuple (min, max) for random delay. Delay is only applied after the first continue.
            on_error: Error handling strategy: "raise" (default) or "return_partial".
            on_error_callback: Optional error callback function with signature:
                (error: Exception, partial_result: ChatResult) -> dict
            original_prompt: Original user prompt (used for dynamic continue_prompt generation).
            **params: Additional parameters to pass to continue requests.

        Returns:
            StreamingIterator: Iterator that yields ChatStreamChunk objects for
                all continuation requests. Access accumulated result via iterator.result.
                The result contains merged text from all continues.

        Raises:
            ValueError: If last_result.finish_reason != "length" or history is not provided.
            Exception: If on_error="raise" and an error occurs during continuation.

        Examples:
            Basic usage:
            >>> history = ChatHistory()
            >>> result = chat("Write a long story", history=history, max_tokens=50)
            >>> if result.finish_reason == "length":
            ...     iterator = ChatContinue.continue_request_stream(chat, result, history=history)
            ...     for chunk in iterator:
            ...         print(chunk.delta, end="", flush=True)
            ...     full_result = iterator.result.to_chat_result()

            With progress tracking:
            >>> def on_progress(count, max_count, current, all_results):
            ...     print(f"继续生成 {count}/{max_count}...")
            >>> iterator = ChatContinue.continue_request_stream(
            ...     chat, result, history=history,
            ...     on_progress=on_progress
            ... )
        """
        if last_result.finish_reason != "length":
            raise ValueError(
                f"continue_request_stream requires finish_reason='length', "
                f"got '{last_result.finish_reason}'"
            )

        if history is None:
            raise ValueError(
                "History is required. Provide history explicitly when calling continue_request_stream."
            )

        # Create working history (immutable - clone original)
        working_history = history.clone()

        # Get original prompt from history if not provided
        if original_prompt is None:
            history_messages = working_history.get_messages()
            for msg in reversed(history_messages):
                if msg.get("role") == "user":
                    original_prompt = msg.get("content", "")
                    break

        # Create generator that yields chunks and tracks results
        all_results: list[ChatResult] = [last_result]

        def _continue_chunk_generator() -> Iterator[ChatStreamChunk]:
            """Generator that yields chunks from all continue requests."""
            nonlocal all_results
            current_result = last_result
            continue_count = 0
            accumulated_text = last_result.text

            while current_result.finish_reason == "length" and continue_count < max_continues:
                continue_count += 1

                # Apply delay if needed (not for first continue)
                ChatContinue._apply_continue_delay(continue_delay, continue_count)

                # Call progress callback
                ChatContinue._call_progress_callback(
                    on_progress, continue_count, max_continues, current_result, all_results
                )

                try:
                    # Get continue prompt (supports string or callable)
                    prompt = ChatContinue._get_continue_prompt(
                        continue_prompt,
                        continue_count,
                        max_continues,
                        accumulated_text,
                        original_prompt or "",
                    )

                    # Add continue prompt if needed
                    if add_continue_prompt:
                        working_history.add_user(prompt)

                    # Stream continue request
                    continue_iterator = chat.stream(
                        working_history.get_messages(), history=working_history, **params
                    )

                    # Yield all chunks from this continue request
                    yield from continue_iterator

                    # Get continue result for next iteration
                    continue_result = continue_iterator.result.to_chat_result()
                    all_results.append(continue_result)
                    current_result = continue_result
                    accumulated_text += continue_result.text

                except Exception as e:
                    # Handle error based on strategy
                    try:
                        ChatContinue._handle_continue_error(
                            e, current_result, all_results, on_error, on_error_callback
                        )
                        # If returning partial, stop iteration
                        break
                    except Exception:
                        # Re-raise if strategy is "raise"
                        raise

        # Create StreamingIterator with custom result that merges all continues
        class MergedContinueIterator(StreamingIterator):
            """Iterator that merges results from all continue requests."""

            def __init__(
                self,
                chunk_gen: Iterator[ChatStreamChunk],
                initial_result: ChatResult,
                all_results_ref: list[ChatResult],
            ):
                super().__init__(chunk_gen)
                self._initial_result = initial_result
                self._all_results_ref = all_results_ref
                self._merged_result: StreamingResult | None = None

            def __iter__(self) -> Iterator[ChatStreamChunk]:
                """Iterate chunks."""
                # Consume the generator to populate all_results
                for chunk in self._iterator:
                    self._result.update(chunk)
                    yield chunk
                # After iteration, ensure all_results is populated correctly
                # Filter out any empty results that might have been added
                if self._all_results_ref:
                    # Remove any empty results (text='' and finish_reason=None)
                    self._all_results_ref[:] = [
                        r
                        for r in self._all_results_ref
                        if not (r.text == "" and r.finish_reason is None)
                    ]

            @property
            def result(self) -> StreamingResult:
                """Get merged result from all continues."""
                if self._merged_result is None:
                    # Merge all results
                    if len(self._all_results_ref) > 1:
                        merged = ChatContinue.merge_results(*self._all_results_ref)
                        self._merged_result = StreamingResult()
                        self._merged_result._text = merged.text
                        self._merged_result._finish_reason = merged.finish_reason
                        self._merged_result._usage = merged.usage
                        self._merged_result._done = True
                    else:
                        # Only initial result, convert to StreamingResult
                        self._merged_result = StreamingResult()
                        self._merged_result._text = self._initial_result.text
                        self._merged_result._finish_reason = self._initial_result.finish_reason
                        self._merged_result._usage = self._initial_result.usage
                        self._merged_result._done = True
                return self._merged_result

        return MergedContinueIterator(_continue_chunk_generator(), last_result, all_results)
