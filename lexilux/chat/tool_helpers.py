"""
Helper functions for tool calling execution and workflow.

This module provides utilities to help with the function calling workflow,
including executing tool calls and managing conversation history with tool calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lexilux.chat.models import ChatResult, ToolCall


def execute_tool_calls(
    result: ChatResult,
    functions: dict[str, Callable],
) -> list[dict[str, Any]]:
    """
    Execute tool calls from a chat result.

    Takes a ChatResult that contains tool calls and executes them using
    the provided functions dictionary. Returns tool response messages
    that can be sent back to the model.

    Args:
        result: ChatResult containing tool calls.
        functions: Dictionary mapping function names to callable functions.
            The function signature should match the arguments in the tool call.

    Returns:
        List of tool response messages (role="tool") with results.

    Raises:
        ValueError: If an unknown function name is encountered.

    Examples:
        >>> def get_weather(location: str, units: str = "celsius") -> str:
        ...     return f"Weather in {location}: 22°{units}"
        >>>
        >>> result = chat("What's the weather in Paris?", tools=[tool])
        >>> tool_responses = execute_tool_calls(
        ...     result,
        ...     {"get_weather": get_weather}
        ... )
        >>>
        >>> # Continue conversation with tool results
        >>> final_result = chat(
        ...     messages=history + tool_responses,
        ...     tools=[tool]
        ... )
    """
    if not result.has_tool_calls:
        return []

    messages: list[dict[str, Any]] = []

    for tool_call in result.tool_calls:
        func = functions.get(tool_call.name)
        if func is None:
            raise ValueError(f"Unknown function: {tool_call.name}")

        # Parse arguments
        try:
            args = tool_call.get_arguments()
        except Exception as e:
            raise ValueError(f"Invalid arguments for {tool_call.name}: {e}") from e

        # Execute function
        try:
            output = func(**args)
        except Exception as e:
            # Return error message to model
            output = f"Error executing {tool_call.name}: {e}"

        # Create tool response message
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.call_id,
            "content": str(output),
        })

    return messages


def create_conversation_history(
    original_messages: list[dict[str, Any]],
    tool_result: ChatResult,
    tool_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create complete conversation history including tool calls and responses.

    Takes the original messages, the assistant result with tool calls,
    and the tool execution outputs, and combines them into a complete
    conversation history ready for the next API request.

    Args:
        original_messages: The messages that led to the tool calls.
        tool_result: The ChatResult containing tool calls.
        tool_outputs: Tool response messages from execute_tool_calls().

    Returns:
        Complete conversation history including:
        - Original messages
        - Assistant message with tool_calls
        - Tool response messages

    Examples:
        >>> messages = [{"role": "user", "content": "What's the weather?"}]
        >>> result = chat(messages, tools=[get_weather_tool])
        >>> if result.has_tool_calls:
        ...     tool_outputs = execute_tool_calls(result, {"get_weather": get_weather})
        ...     history = create_conversation_history(messages, result, tool_outputs)
        ...     final_result = chat(history, tools=[get_weather_tool])
    """
    # Copy original messages
    history = list(original_messages)

    # Add assistant message with tool_calls
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": tool_result.text,
    }

    # Add tool_calls if present
    if tool_result.has_tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in tool_result.tool_calls
        ]

    history.append(assistant_message)

    # Add tool response messages
    history.extend(tool_outputs)

    return history


class ToolCallHelper:
    """
    Helper class for managing tool calling workflows.

    Provides a higher-level abstraction for common tool calling patterns
    including automatic execution and conversation continuation.

    Attributes:
        functions: Dictionary mapping function names to callables.

    Examples:
        >>> def get_weather(location: str) -> str:
        ...     return f"Weather in {location}: 22°C"
        >>>
        >>> helper = ToolCallHelper({"get_weather": get_weather})
        >>>
        >>> result = chat("What's the weather in Paris?", tools=[tool])
        >>> if result.has_tool_calls:
        ...     final_result = helper.continue_conversation(
        ...         chat=chat,
        ...         messages=[{"role": "user", "content": "..."}],
        ...         tool_result=result,
        ...         tools=[tool]
        ...     )
    """

    def __init__(self, functions: dict[str, Callable]):
        """
        Initialize ToolCallHelper.

        Args:
            functions: Dictionary mapping function names to callable functions.
        """
        self.functions = functions

    def execute_tool_calls(self, result: ChatResult) -> list[dict[str, Any]]:
        """
        Execute tool calls from a chat result.

        Args:
            result: ChatResult containing tool calls.

        Returns:
            List of tool response messages.
        """
        return execute_tool_calls(result, self.functions)

    def continue_conversation(
        self,
        chat: Any,  # Chat instance (avoid circular import)
        messages: list[dict[str, Any]],
        tool_result: ChatResult,
        tools: list[Any] | None = None,
    ) -> ChatResult:
        """
        Continue conversation after tool execution.

        Executes tool calls, builds conversation history, and sends
        a follow-up request to get the final response.

        Args:
            chat: Chat client instance.
            messages: Original messages that led to tool calls.
            tool_result: ChatResult containing tool calls.
            tools: Tools to pass to follow-up request.

        Returns:
            Final ChatResult after tool execution.

        Examples:
            >>> helper = ToolCallHelper({"get_weather": get_weather})
            >>> result = chat("What's the weather?", tools=[tool])
            >>> if result.has_tool_calls:
            ...     final = helper.continue_conversation(
            ...         chat=chat,
            ...         messages=[{"role": "user", "content": "What's the weather?"}],
            ...         tool_result=result,
            ...         tools=[tool]
            ...     )
            >>> print(final.text)
        """
        # Execute tool calls
        tool_outputs = self.execute_tool_calls(tool_result)

        # Build complete history
        history = create_conversation_history(messages, tool_result, tool_outputs)

        # Send follow-up request
        return chat(messages=history, tools=tools)
