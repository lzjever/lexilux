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
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, ChatStreamChunk, MessagesLike, ToolCall
from lexilux.chat.params import ChatParams
from lexilux.chat.utils import normalize_finish_reason, normalize_messages, parse_usage
from lexilux.usage import Json, Usage

if TYPE_CHECKING:
    from lexilux.chat.tools import Tool

DEFAULT_TEMPERATURE = 0.7


def prepare_messages_for_request(
    messages: MessagesLike,
    *,
    system: str | None,
    history: ChatHistory | None,
) -> tuple[list[dict[str, Any]], ChatHistory | None, list[Any]]:
    base_messages = normalize_messages(messages, system=system)
    user_messages_to_add: list[Any] = [
        msg.get("content", "") for msg in base_messages if msg.get("role") == "user"
    ]

    if history is None:
        return base_messages, None, user_messages_to_add

    working_history = history.clone()
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
        param_dict = {"temperature": DEFAULT_TEMPERATURE if temperature is None else temperature}

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

    tool_calls_list = _parse_tool_calls(message.get("tool_calls"))
    finish_reason = normalize_finish_reason(choice.get("finish_reason"))
    usage = parse_usage(response_data)

    return ChatResult(
        text=text,
        usage=usage,
        finish_reason=finish_reason,
        tool_calls=tool_calls_list,
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


class SSEChatStreamParser:
    def __init__(self, *, return_raw_events: bool):
        self._return_raw_events = return_raw_events
        self._final_usage: Usage | None = None
        self._final_finish_reason: str | None = None
        self._done = False

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
            final_usage = self._final_usage if self._final_usage is not None else Usage()
            return ChatStreamChunk(
                delta="",
                done=True,
                usage=final_usage,
                finish_reason=self._final_finish_reason,
                raw={"done": True} if self._return_raw_events else {},
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
        tool_calls_list = _parse_stream_tool_calls(delta.get("tool_calls"))

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

        return ChatStreamChunk(
            delta=content,
            done=done,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls_list,
            raw=event_data if self._return_raw_events else {},
        )
