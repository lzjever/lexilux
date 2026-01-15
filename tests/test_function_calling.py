"""
Unit tests for function calling and multimodal support.

Tests the new tool calling and multimodal content features.
"""

import pytest

from lexilux.chat import (
    ContentBlock,
    FunctionTool,
    ImageContentBlock,
    ImageUrlDetail,
    TextContentBlock,
    ToolCall,
    ToolChoice,
    ToolCallHelper,
    create_conversation_history,
    execute_tool_calls,
    normalize_messages,
)
from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.chat.params import ChatParams
from lexilux.usage import Usage


class TestContentBlocks:
    """Tests for content block types."""

    def test_text_content_block(self):
        """Test TextContentBlock structure."""
        block: TextContentBlock = {"type": "text", "text": "Hello, world!"}
        assert block["type"] == "text"
        assert block["text"] == "Hello, world!"

    def test_image_content_block(self):
        """Test ImageContentBlock structure."""
        block: ImageContentBlock = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/image.jpg"

    def test_image_content_block_with_detail(self):
        """Test ImageContentBlock with detail parameter."""
        block: ImageContentBlock = {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg",
                "detail": "high"
            }
        }
        assert block["image_url"]["detail"] == "high"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tc = ToolCall(
            id="call_123",
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "Paris"}'
        )
        assert tc.id == "call_123"
        assert tc.call_id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == '{"location": "Paris"}'

    def test_tool_call_get_arguments(self):
        """Test parsing arguments from ToolCall."""
        tc = ToolCall(
            id="call_123",
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "Paris", "units": "celsius"}'
        )
        args = tc.get_arguments()
        assert args == {"location": "Paris", "units": "celsius"}

    def test_tool_call_to_dict(self):
        """Test converting ToolCall to API format."""
        tc = ToolCall(
            id="call_123",
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "Paris"}'
        )
        result = tc.to_dict()
        assert result["id"] == "call_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["arguments"] == '{"location": "Paris"}'


class TestFunctionTool:
    """Tests for FunctionTool dataclass."""

    def test_function_tool_creation(self):
        """Test creating a FunctionTool."""
        tool = FunctionTool(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )
        assert tool.name == "get_weather"
        assert tool.type == "function"

    def test_function_tool_to_dict(self):
        """Test converting FunctionTool to API format."""
        tool = FunctionTool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {}}
        )
        result = tool.to_dict()
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"

    def test_function_tool_with_strict(self):
        """Test FunctionTool with strict mode."""
        tool = FunctionTool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {}},
            strict=True
        )
        result = tool.to_dict()
        assert result["strict"] is True


class TestToolChoice:
    """Tests for ToolChoice dataclass."""

    def test_tool_choice_auto(self):
        """Test ToolChoice with auto mode."""
        choice = ToolChoice(type="auto")
        assert choice.to_dict() == "auto"

    def test_tool_choice_required(self):
        """Test ToolChoice with required mode."""
        choice = ToolChoice(type="required")
        assert choice.to_dict() == "required"

    def test_tool_choice_function(self):
        """Test ToolChoice with specific function."""
        choice = ToolChoice(type="function", name="get_weather")
        result = choice.to_dict()
        assert result["type"] == "function"
        assert result["name"] == "get_weather"


class TestChatParams:
    """Tests for ChatParams with tools."""

    def test_chat_params_with_tools(self):
        """Test ChatParams with tools."""
        tool = FunctionTool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}}
        )
        params = ChatParams(tools=[tool])
        result = params.to_dict()
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"

    def test_chat_params_with_tool_choice(self):
        """Test ChatParams with tool_choice."""
        params = ChatParams(tool_choice="required")
        result = params.to_dict()
        assert result["tool_choice"] == "required"

    def test_chat_params_with_parallel_tool_calls(self):
        """Test ChatParams with parallel_tool_calls."""
        params = ChatParams(parallel_tool_calls=False)
        result = params.to_dict()
        assert result["parallel_tool_calls"] is False


class TestChatResult:
    """Tests for ChatResult with tool_calls."""

    def test_chat_result_with_tool_calls(self):
        """Test ChatResult with tool_calls."""
        tool_call = ToolCall(
            id="call_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}'
        )
        result = ChatResult(
            text="",
            usage=Usage(),
            tool_calls=[tool_call]
        )
        assert result.has_tool_calls is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

    def test_chat_result_has_tool_calls_property(self):
        """Test has_tool_calls property."""
        # With tool calls
        tc = ToolCall(id="1", call_id="1", name="test", arguments="{}")
        result = ChatResult(text="", usage=Usage(), tool_calls=[tc])
        assert result.has_tool_calls is True

        # Without tool calls
        result = ChatResult(text="Hello", usage=Usage())
        assert result.has_tool_calls is False


class TestChatStreamChunk:
    """Tests for ChatStreamChunk with tool_calls."""

    def test_stream_chunk_with_tool_calls(self):
        """Test ChatStreamChunk with tool_calls."""
        tc = ToolCall(id="1", call_id="1", name="test", arguments="{}")
        chunk = ChatStreamChunk(
            delta="",
            usage=Usage(),
            done=False,
            tool_calls=[tc]
        )
        assert chunk.has_tool_calls is True
        assert chunk.has_content is False

    def test_stream_chunk_has_content_property(self):
        """Test has_content property."""
        # With content
        chunk = ChatStreamChunk(delta="Hello", usage=Usage(), done=False)
        assert chunk.has_content is True

        # Without content
        chunk = ChatStreamChunk(delta="", usage=Usage(), done=False)
        assert chunk.has_content is False


class TestNormalizeMessages:
    """Tests for normalize_messages with multimodal support."""

    def test_normalize_simple_string(self):
        """Test normalizing a simple string (backward compatibility)."""
        result = normalize_messages("Hello")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_normalize_multimodal_content(self):
        """Test normalizing multimodal content."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }]
        result = normalize_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    def test_normalize_validates_content_blocks(self):
        """Test that content blocks are validated."""
        # Missing type field
        messages = [{
            "role": "user",
            "content": [{"text": "Hello"}]  # Missing "type"
        }]
        with pytest.raises(ValueError, match="must have a 'type' key"):
            normalize_messages(messages)

    def test_normalize_mixed_string_and_multimodal(self):
        """Test mixing string and multimodal messages."""
        messages = [
            {"role": "user", "content": "First message"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Second message"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]
            }
        ]
        result = normalize_messages(messages)
        assert len(result) == 2
        assert isinstance(result[0]["content"], str)
        assert isinstance(result[1]["content"], list)


class TestToolHelpers:
    """Tests for tool helper functions."""

    def test_execute_tool_calls(self):
        """Test executing tool calls."""
        def get_weather(location: str) -> str:
            return f"Weather in {location}: 22°C"

        tool_call = ToolCall(
            id="call_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}'
        )

        result = ChatResult(
            text="",
            usage=Usage(),
            tool_calls=[tool_call]
        )

        responses = execute_tool_calls(result, {"get_weather": get_weather})
        assert len(responses) == 1
        assert responses[0]["role"] == "tool"
        assert responses[0]["tool_call_id"] == "call_1"
        assert "Paris" in responses[0]["content"]

    def test_execute_tool_calls_unknown_function(self):
        """Test executing tool calls with unknown function."""
        tool_call = ToolCall(
            id="call_1",
            call_id="call_1",
            name="unknown_func",
            arguments="{}"
        )

        result = ChatResult(
            text="",
            usage=Usage(),
            tool_calls=[tool_call]
        )

        with pytest.raises(ValueError, match="Unknown function"):
            execute_tool_calls(result, {})

    def test_create_conversation_history(self):
        """Test creating complete conversation history."""
        tool_call = ToolCall(
            id="call_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}'
        )

        tool_result = ChatResult(
            text="I'll check the weather for you.",
            usage=Usage(),
            tool_calls=[tool_call]
        )

        tool_outputs = [
            {"role": "tool", "tool_call_id": "call_1", "content": "22°C in Paris"}
        ]

        original_messages = [
            {"role": "user", "content": "What's the weather in Paris?"}
        ]

        history = create_conversation_history(
            original_messages,
            tool_result,
            tool_outputs
        )

        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "tool_calls" in history[1]
        assert history[2]["role"] == "tool"


class TestToolCallHelper:
    """Tests for ToolCallHelper class."""

    def test_tool_call_helper_init(self):
        """Test initializing ToolCallHelper."""
        def dummy_func() -> str:
            return "result"

        helper = ToolCallHelper({"dummy": dummy_func})
        assert helper.functions == {"dummy": dummy_func}

    def test_tool_call_helper_execute(self):
        """Test ToolCallHelper.execute_tool_calls."""
        def get_weather(location: str) -> str:
            return f"Weather in {location}: 22°C"

        tool_call = ToolCall(
            id="call_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}'
        )

        result = ChatResult(
            text="",
            usage=Usage(),
            tool_calls=[tool_call]
        )

        helper = ToolCallHelper({"get_weather": get_weather})
        responses = helper.execute_tool_calls(result)

        assert len(responses) == 1
        assert responses[0]["content"] == "Weather in Paris: 22°C"
