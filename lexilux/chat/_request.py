"""
Shared request/response plumbing for the chat client.

This module centralizes:
- message/history preparation
- parameter normalization (ChatParams + keyword overrides)
- response parsing (non-stream)
- SSE stream parsing (stream)
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import (
    ChatResult,
    ChatStreamChunk,
    MessagesLike,
    StreamingToolCall,
    ToolCall,
)
from lexilux.chat.params import ChatParams
from lexilux.chat.utils import normalize_finish_reason, normalize_messages, parse_usage
from lexilux.usage import Json, Usage

if TYPE_CHECKING:
    from lexilux.chat.tools import Tool

DEFAULT_TEMPERATURE = 0.7


def build_api_messages(
    messages: MessagesLike,
    *,
    system: str | None,
    history: ChatHistory | None,
) -> list[dict[str, Any]]:
    """
    Build messages list for API request (read-only from history).

    This is a pure function that only reads from history, never modifies it.
    No cloning is needed.

    Args:
        messages: Input messages.
        system: Optional system message.
        history: Optional chat history (read-only).

    Returns:
        Combined messages list ready for API.
    """
    base_messages = normalize_messages(messages, system=system)

    if history is None:
        return base_messages

    # Read-only: just get messages, no clone needed
    history_messages = history.get_messages(include_system=True)
    return history_messages + base_messages


def prepare_messages_for_request(
    messages: MessagesLike,
    *,
    system: str | None,
    history: ChatHistory | None,
    clone_history: bool = True,
) -> tuple[list[dict[str, Any]], ChatHistory | None, list[Any]]:
    """
    Prepare messages for API request with mutable history tracking.

    This version is for complete() family methods that need to track
    conversation state across multiple API calls.

    Args:
        messages: Input messages.
        system: Optional system message.
        history: Optional chat history.
        clone_history: Whether to clone history (default True for safety).
            Set to False when caller has already created a working copy.

    Returns:
        Tuple of (normalized_messages, working_history, user_messages_to_add).
    """
    base_messages = normalize_messages(messages, system=system)
    user_messages_to_add: list[Any] = [
        msg.get("content", "") for msg in base_messages if msg.get("role") == "user"
    ]

    if history is None:
        return base_messages, None, user_messages_to_add

    working_history = history.clone() if clone_history else history
    history_messages = working_history.get_messages(include_system=True)
    return history_messages + base_messages, working_history, user_messages_to_add


def _normalize_stop(stop: str | Sequence[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def build_params_dict(
    *,
    params: ChatParams | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    stop: str | Sequence[str] | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    logit_bias: dict[int, float] | None,
    user: str | None,
    n: int | None = None,
    tools: list[Tool] | None,
    tool_choice: str | Any | None,
    parallel_tool_calls: bool | None,
) -> Json:
    if params is not None:
        param_dict: Json = params.to_dict(exclude_none=True)
    else:
        param_dict = {
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature
        }

    if temperature is not None:
        param_dict["temperature"] = temperature
    if top_p is not None:
        param_dict["top_p"] = top_p
    if max_tokens is not None:
        param_dict["max_tokens"] = max_tokens
    normalized_stop = _normalize_stop(stop)
    if normalized_stop is not None:
        param_dict["stop"] = normalized_stop
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

    if tools is not None:
        param_dict["tools"] = [tool.to_dict() for tool in tools]

    if tool_choice is not None:
        if isinstance(tool_choice, str):
            param_dict["tool_choice"] = tool_choice
        else:
            to_dict = getattr(tool_choice, "to_dict", None)
            param_dict["tool_choice"] = to_dict() if callable(to_dict) else tool_choice

    if parallel_tool_calls is not None:
        param_dict["parallel_tool_calls"] = parallel_tool_calls

    return param_dict


def build_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    params: Json,
    stream: bool,
    include_usage: bool,
    extra: Json | None,
) -> Json:
    payload: Json = {"model": model, "messages": messages, **params}
    if stream:
        payload["stream"] = True
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
    if extra:
        payload.update(extra)
    return payload


def parse_chat_completion_response(
    response_data: Json,
    *,
    return_raw: bool,
) -> ChatResult:
    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError("No choices in API response")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError(f"Invalid choice format: expected dict, got {type(choice)}")

    message = choice.get("message", {})
    if not isinstance(message, dict):
        message = {}
    text = message.get("content", "") or ""

    # Extract reasoning content (for DeepSeek, Kimi, GLM, etc.)
    reasoning = message.get("reasoning_content") or None

    # For Anthropic-style responses with content blocks
    if not reasoning:
        content_blocks = message.get("content", [])
        if isinstance(content_blocks, list):
            reasoning_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    reasoning_parts.append(block.get("thinking", ""))
            if reasoning_parts:
                reasoning = "\n".join(reasoning_parts)

    tool_calls_list = _parse_tool_calls(message.get("tool_calls"))

    # Side patch: fallback to text-based tool call parsing for flawed providers
    if not tool_calls_list and text:
        tool_calls_list = _parse_text_tool_calls(text)

    finish_reason = normalize_finish_reason(choice.get("finish_reason"))
    usage = parse_usage(response_data)

    return ChatResult(
        text=text,
        usage=usage,
        finish_reason=finish_reason,
        tool_calls=tool_calls_list,
        reasoning=reasoning,
        raw=response_data if return_raw else {},
    )


def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        if not isinstance(tc, dict):
            continue
        try:
            function = tc.get("function", {})
            if not isinstance(function, dict):
                function = {}
            call_id = str(tc.get("id", ""))
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    call_id=call_id,
                    name=str(function.get("name", "")),
                    arguments=str(function.get("arguments", "{}")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tool_calls


def _parse_stream_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        if not isinstance(tc, dict):
            continue
        try:
            function = tc.get("function", {})
            if not isinstance(function, dict):
                function = {}
            call_id = tc.get("id", "")
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    call_id=call_id,
                    name=str(function.get("name", "")),
                    arguments=str(function.get("arguments", "{}")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tool_calls


def _parse_text_tool_calls(text: str) -> list[ToolCall]:
    """
    Side patch: Parse tool calls from text format for flawed API providers.

    Handles formats like: <tool_call>function_name</tool_call>
    This is a non-invasive fallback for providers that don't follow OpenAI format.
    """
    if not text:
        return []

    tool_calls: list[ToolCall] = []

    # Pattern to match <tool_call>function_name</tool_call>
    # Also handles potential arguments inside
    pattern = r"<tool_call>\s*([^<\n]+)(?:\n\n?([^<]*?))?\s*</tool_call>"

    matches = re.findall(pattern, text, re.DOTALL)

    for i, match in enumerate(matches):
        function_name = match[0].strip()
        arguments_text = match[1].strip() if len(match) > 1 and match[1] else "{}"

        # Generate a simple call ID
        call_id = f"text_call_{i + 1}"

        if function_name:
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    call_id=call_id,
                    name=function_name,
                    arguments=arguments_text if arguments_text else "{}",
                )
            )

    return tool_calls


class SSEChatStreamParser:
    """
    Parser for Server-Sent Events (SSE) chat streaming responses.

    Handles incremental parsing of streaming responses including:
    - Text content deltas
    - Tool call accumulation (streaming tool calls)
    - Reasoning content (for thinking models)
    - Usage statistics
    """

    def __init__(self, *, return_raw_events: bool, include_reasoning: bool = False):
        self._return_raw_events = return_raw_events
        self._include_reasoning = include_reasoning
        self._final_usage: Usage | None = None
        self._final_finish_reason: str | None = None
        self._done = False

        # Tool call accumulation state
        # key: index (0, 1, 2...), value: {"id": str, "name": str, "arguments": str}
        self._tool_call_state: dict[int, dict[str, str]] = {}

    @property
    def done(self) -> bool:
        return self._done

    def feed_line(self, line: str) -> ChatStreamChunk | None:
        if self._done:
            return None

        if not line.startswith("data: "):
            return None

        data_str = line[6:]
        if data_str == "[DONE]":
            self._done = True
            final_usage = (
                self._final_usage if self._final_usage is not None else Usage()
            )
            return ChatStreamChunk(
                delta="",
                done=True,
                usage=final_usage,
                finish_reason=self._final_finish_reason,
                raw={"done": True} if self._return_raw_events else {},
                reasoning_content=None,
                reasoning_tokens=None,
            )

        try:
            event_data = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        choices = event_data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        if not isinstance(choice, dict):
            return None

        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}
        content = delta.get("content") or ""

        # Process streaming tool calls with accumulation
        raw_tool_calls = delta.get("tool_calls", [])
        streaming_tool_calls = self._process_tool_call_deltas(raw_tool_calls)

        # Extract complete tool calls (those with valid JSON arguments)
        complete_tool_calls = []
        for stc in streaming_tool_calls:
            if stc.is_complete:
                # Get id and name from accumulated state
                state = self._tool_call_state.get(stc.index, {})
                tc_id = state.get("id", "")
                tc_name = state.get("name", "")
                if tc_id and tc_name:
                    complete_tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            call_id=tc_id,
                            name=tc_name,
                            arguments=stc.arguments_accumulated,
                        )
                    )

        finish_reason = normalize_finish_reason(choice.get("finish_reason"))
        done = finish_reason is not None
        if finish_reason is not None:
            self._final_finish_reason = finish_reason

        if "usage" in event_data:
            usage = parse_usage(event_data)
            self._final_usage = usage
        elif done and self._final_usage is None:
            usage = Usage()
            self._final_usage = usage
        else:
            usage = Usage()

        # Parse reasoning fields if enabled
        reasoning_content = None
        reasoning_tokens = None
        if self._include_reasoning:
            if "reasoning_content" in delta:
                reasoning_content = delta["reasoning_content"]
            if "reasoning_tokens" in delta:
                reasoning_tokens = delta["reasoning_tokens"]

        return ChatStreamChunk(
            delta=content,
            done=done,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=complete_tool_calls,
            streaming_tool_calls=streaming_tool_calls,
            raw=event_data if self._return_raw_events else {},
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
        )

    def _process_tool_call_deltas(
        self, raw_tool_calls: list[Any]
    ) -> list[StreamingToolCall]:
        """
        Process streaming tool call deltas and accumulate state.

        OpenAI streaming format:
        - First chunk: {"index": 0, "id": "call_xxx", "function": {"name": "write", "arguments": ""}}
        - Later chunks: {"index": 0, "function": {"arguments": "{\\"file_path\\":"}}

        Args:
            raw_tool_calls: Raw tool_calls array from delta

        Returns:
            List of StreamingToolCall objects with accumulated state
        """
        if not isinstance(raw_tool_calls, list):
            return []

        results: list[StreamingToolCall] = []

        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue

            index = tc.get("index", 0)
            function = tc.get("function", {}) or {}

            # Check if this is the first chunk for this tool call
            is_first = index not in self._tool_call_state

            if is_first:
                # Initialize state for this tool call
                self._tool_call_state[index] = {
                    "id": tc.get("id") or "",
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "",
                }
            else:
                # Accumulate arguments
                args_delta = function.get("arguments") or ""
                self._tool_call_state[index]["arguments"] += args_delta

            state = self._tool_call_state[index]
            args_delta = function.get("arguments") or ""

            # Check if accumulated arguments form valid JSON
            is_complete = self._is_valid_json(state["arguments"])

            results.append(
                StreamingToolCall(
                    index=index,
                    id=state["id"] if is_first else None,
                    name=state["name"] if is_first else None,
                    arguments_delta=args_delta,
                    arguments_accumulated=state["arguments"],
                    is_first=is_first,
                    is_complete=is_complete,
                )
            )

        return results

    @staticmethod
    def _is_valid_json(s: str) -> bool:
        """Check if string is valid JSON."""
        if not s:
            return False
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False
