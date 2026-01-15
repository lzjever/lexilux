"""
Tool definitions for function calling support.

This module defines types for tools that can be passed to the chat API,
enabling function calling capabilities where models can request execution
of specific functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class FunctionTool:
    """
    Function tool definition.

    Represents a function that the model can call during chat completion.

    Attributes:
        name: Function name (must be unique and match a callable function).
        description: Description of when and how to use the function.
        parameters: JSON Schema defining the function's input arguments.
        strict: Whether to enforce strict schema adherence (default: False).
        type: Tool type, always "function".

    Examples:
        >>> tool = FunctionTool(
        ...     name="get_weather",
        ...     description="Get current weather for a location",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {
        ...                 "type": "string",
        ...                 "description": "City name, e.g. Paris"
        ...             }
        ...         },
        ...         "required": ["location"]
        ...     }
        ... )
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    type: Literal["function"] = "function"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to API request format.

        Returns:
            Dictionary in OpenAI tool format.

        Examples:
            >>> tool = FunctionTool(name="get_weather", description="...", parameters={})
            >>> tool.to_dict()
            {'type': 'function', 'name': 'get_weather', 'description': '...', 'parameters': {}, 'strict': False}
        """
        result: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "description": self.description,
        }

        if self.parameters:
            result["parameters"] = self.parameters

        if self.strict:
            result["strict"] = True

        return result


# Type alias for all tool types
Tool = FunctionTool


@dataclass
class ToolChoice:
    """
    Tool choice strategy configuration.

    Controls when and how the model uses tools.

    Attributes:
        type: Choice strategy type.
            - "auto": Model decides whether to call tools (default).
            - "required": Model must call one or more tools.
            - "function": Must call a specific function.
            - "allowed_tools": Restrict to a subset of tools.
        name: Function name (required when type="function").
        tools: List of allowed tools (required when type="allowed_tools").

    Examples:
        >>> # Auto mode (let model decide)
        >>> choice = ToolChoice(type="auto")
        >>>
        >>> # Require tool calls
        >>> choice = ToolChoice(type="required")
        >>>
        >>> # Force specific function
        >>> choice = ToolChoice(type="function", name="get_weather")
        >>>
        >>> # Restrict to specific tools
        >>> choice = ToolChoice(
        ...     type="allowed_tools",
        ...     tools=[FunctionTool(name="get_weather", ...)]
        ... )
    """
    type: Literal["auto", "required", "function", "allowed_tools"]
    name: str | None = None
    tools: list[Tool] | None = None

    def to_dict(self) -> dict[str, Any] | str:
        """
        Convert to API request format.

        Returns:
            String for simple modes, dict for complex modes.

        Examples:
            >>> ToolChoice(type="auto").to_dict()
            'auto'
            >>> ToolChoice(type="required").to_dict()
            'required'
            >>> ToolChoice(type="function", name="get_weather").to_dict()
            {'type': 'function', 'name': 'get_weather'}
        """
        if self.type == "auto":
            return "auto"
        elif self.type == "required":
            return "required"
        elif self.type == "function":
            return {
                "type": "function",
                "name": self.name,
            }
        elif self.type == "allowed_tools":
            return {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": tool.name} for tool in (self.tools or [])
                ],
            }
        else:
            # Fallback
            return self.type


# Type alias for tool choice parameter
ToolChoiceParam = str | ToolChoice | None
