"""
Continue functionality for chat completions.

Provides ChatContinue class for handling continuation requests when generation
is stopped due to max_tokens limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Literal, overload

from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.chat.streaming import StreamingIterator, StreamingResult
from lexilux.usage import Usage

if TYPE_CHECKING:
    from lexilux.chat.client import Chat


class ChatContinue:
    """
    Continue functionality handler (user is responsible for determining if continue is needed).

    This class provides utilities for continuing generation when finish_reason == "length".
    The user must check finish_reason and decide when to continue.
    """

    @staticmethod
    @overload
    def continue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        history: ChatHistory | None = None,
        add_continue_prompt: bool = True,
        continue_prompt: str = "continue",
        max_continues: int = 1,
        auto_merge: Literal[True] = True,
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
        continue_prompt: str = "continue",
        max_continues: int = 1,
        auto_merge: Literal[False],
        **params: Any,
    ) -> list[ChatResult]: ...

    @staticmethod
    def continue_request(
        chat: Chat,
        last_result: ChatResult,
        *,
        history: ChatHistory | None = None,
        add_continue_prompt: bool = True,
        continue_prompt: str = "continue",
        max_continues: int = 1,
        auto_merge: bool = True,
        **params: Any,
    ) -> ChatResult | list[ChatResult]:
        """
        Continue generation request (enhanced version).

        Automatically handles continuation when finish_reason == "length", with support
        for multiple continues and automatic merging.

        Args:
            chat: Chat client instance.
            last_result: Last result (must have finish_reason == "length").
            history: Conversation history (required). Must be provided explicitly.
            add_continue_prompt: Whether to add a user continue instruction round.
            continue_prompt: User prompt when add_continue_prompt=True.
            max_continues: Maximum number of continuation attempts. If result is still
                truncated after max_continues, returns merged result (if auto_merge=True)
                or list of results (if auto_merge=False).
            auto_merge: If True, automatically merge all results into a single ChatResult.
                If False, returns a list of all results [last_result, continue_result1, ...].
            **params: Additional parameters to pass to chat (temperature, max_tokens, etc.).

        Returns:
            If auto_merge=True: Merged ChatResult with all continuation results combined.
            If auto_merge=False: List of ChatResult instances [last_result, continue_result1, ...].

        Raises:
            ValueError: If last_result.finish_reason != "length" or history is not provided.

        Examples:
            Basic usage:
            >>> history = ChatHistory()
            >>> result = chat("Write a long story", history=history, max_tokens=50)
            >>> if result.finish_reason == "length":
            ...     full_result = ChatContinue.continue_request(chat, result, history=history)
            ...     print(full_result.text)  # Complete merged text

            Multiple continues:
            >>> history = ChatHistory()
            >>> result = chat("Very long story", history=history, max_tokens=30)
            >>> if result.finish_reason == "length":
            ...     full_result = ChatContinue.continue_request(chat, result, history=history, max_continues=3)

            Get all intermediate results:
            >>> history = ChatHistory()
            >>> result = chat("Story", history=history, max_tokens=50)
            >>> if result.finish_reason == "length":
            ...     all_results = ChatContinue.continue_request(chat, result, history=history, auto_merge=False)
            ...     # all_results = [result, continue_result1, continue_result2, ...]
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

        all_results = [last_result]
        current_result = last_result
        continue_count = 0

        while current_result.finish_reason == "length" and continue_count < max_continues:
            continue_count += 1

            # Execute single continue request
            if add_continue_prompt:
                history.add_user(continue_prompt)

            continue_result = chat(history.get_messages(), history=history, **params)
            all_results.append(continue_result)
            current_result = continue_result
            # Note: history is automatically updated by chat() call above

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
        continue_prompt: str = "continue",
        max_continues: int = 1,
        **params: Any,
    ) -> StreamingIterator:
        """
        Continue generation with streaming output.

        This is the streaming version of `continue_request()`. It returns a
        StreamingIterator that yields chunks for all continuation requests.

        Args:
            chat: Chat client instance.
            last_result: Last result (must have finish_reason == "length").
            history: Conversation history (required). Must be provided explicitly.
            add_continue_prompt: Whether to add a user continue instruction round.
            continue_prompt: User prompt when add_continue_prompt=True.
            max_continues: Maximum number of continuation attempts. If result is still
                truncated after max_continues, returns merged result.
            **params: Additional parameters to pass to continue requests.

        Returns:
            StreamingIterator: Iterator that yields ChatStreamChunk objects for
                all continuation requests. Access accumulated result via iterator.result.
                The result contains merged text from all continues.

        Raises:
            ValueError: If last_result.finish_reason != "length" or history is not provided.

        Examples:
            Basic usage:
            >>> history = ChatHistory()
            >>> result = chat("Write a long story", history=history, max_tokens=50)
            >>> if result.finish_reason == "length":
            ...     iterator = ChatContinue.continue_request_stream(chat, result, history=history)
            ...     for chunk in iterator:
            ...         print(chunk.delta, end="", flush=True)
            ...     full_result = iterator.result.to_chat_result()
            ...     print(f"\nComplete story: {len(full_result.text)} chars")

            Multiple continues:
            >>> history = ChatHistory()
            >>> result = chat("Very long story", history=history, max_tokens=30)
            >>> if result.finish_reason == "length":
            ...     iterator = ChatContinue.continue_request_stream(
            ...         chat, result, history=history, max_continues=3
            ...     )
            ...     for chunk in iterator:
            ...         print(chunk.delta, end="", flush=True)
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

        # Create generator that yields chunks and tracks results
        all_results: list[ChatResult] = [last_result]

        def _continue_chunk_generator() -> Iterator[ChatStreamChunk]:
            """Generator that yields chunks from all continue requests."""
            nonlocal all_results
            current_result = last_result
            continue_count = 0

            while current_result.finish_reason == "length" and continue_count < max_continues:
                continue_count += 1

                # Add continue prompt if needed
                if add_continue_prompt:
                    history.add_user(continue_prompt)

                # Stream continue request
                continue_iterator = chat.stream(
                    history.get_messages(), history=history, **params
                )

                # Yield all chunks from this continue request
                for chunk in continue_iterator:
                    yield chunk

                # Get continue result for next iteration
                continue_result = continue_iterator.result.to_chat_result()
                all_results.append(continue_result)
                current_result = continue_result
                # Note: history is automatically updated by chat.stream() call above

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
                        r for r in self._all_results_ref
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
