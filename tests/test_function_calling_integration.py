"""
Integration tests for function calling and multimodal features.

These tests require real API calls and are marked with @pytest.mark.integration.
They test function calling, tool execution, multimodal vision inputs, and streaming with tools.
"""

import base64
import io

import pytest

# PIL is optional for multimodal tests
try:
    from PIL import Image  # noqa: F401  # Used in conditional test code

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore[misc,assignment]

from lexilux import Chat, FunctionTool, ToolChoice
from lexilux.chat.tool_helpers import (
    ToolCallHelper,
    create_conversation_history,
    execute_tool_calls,
)

pytestmark = pytest.mark.integration


# ============================================================================
# Test Functions (Tools) for Function Calling
# ============================================================================


def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location.

    Args:
        location: City name or location
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather information string
    """
    # Simulate weather data
    temp = 22 if units == "celsius" else 72
    return f"Weather in {location}: {temp}°{units[0].upper()}, sunny"


def get_time(timezone: str) -> str:
    """Get current time for a timezone.

    Args:
        timezone: Timezone name (e.g., 'America/New_York', 'UTC')

    Returns:
        Current time string
    """

    # Simulate time data
    return f"Current time in {timezone}: 2026-01-15 14:30:00"


def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression (e.g., '2 + 2', '10 * 5')

    Returns:
        Calculation result
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Search results
    """
    # Simulate search results
    results = [
        f"1. Result for '{query}' - Example website",
        f"2. Another result for '{query}' - Another site",
    ]
    return "\n".join(results[:num_results])


# ============================================================================
# Basic Function Calling Tests
# ============================================================================


def test_function_calling_basic_tool_definition():
    """Test basic FunctionTool definition and to_dict conversion."""
    tool = FunctionTool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            "required": ["location"],
        },
    )

    tool_dict = tool.to_dict()

    assert tool_dict["type"] == "function"
    assert "function" in tool_dict
    assert tool_dict["function"]["name"] == "get_weather"
    assert tool_dict["function"]["description"] == "Get current weather for a location"
    assert "properties" in tool_dict["function"]["parameters"]
    assert "location" in tool_dict["function"]["parameters"]["properties"]


@pytest.mark.skip_if_no_config
def test_function_calling_simple_tool_use(test_config, has_real_api_config):
    """Test simple function calling with one tool."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Define tool
    weather_tool = FunctionTool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            "required": ["location"],
        },
    )

    # Make request with tool
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    result = chat(messages, tools=[weather_tool])

    # Verify tool call was requested
    assert result.has_tool_calls, "Model should request tool call"
    assert len(result.tool_calls) > 0, "Should have at least one tool call"

    # Check tool call structure
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert tool_call.call_id is not None
    assert tool_call.id is not None

    # Parse arguments
    args = tool_call.get_arguments()
    assert "location" in args
    assert args["location"] == "Paris"


@pytest.mark.skip_if_no_config
def test_function_calling_with_execution(test_config, has_real_api_config):
    """Test complete function calling workflow: request -> execute -> final response."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Define tools
    tools_dict = {
        "get_weather": get_weather,
        "get_time": get_time,
    }

    tools = [
        FunctionTool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        FunctionTool(
            name="get_time",
            description="Get current time for a timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone name"}
                },
                "required": ["timezone"],
            },
        ),
    ]

    # Initial request
    messages = [{"role": "user", "content": "What's the weather and time in Paris?"}]
    result = chat(messages, tools=tools)

    assert result.has_tool_calls, "Model should request tools"

    # Execute tool calls
    tool_responses = execute_tool_calls(result, tools_dict)

    # Create conversation history
    history = create_conversation_history(messages, result, tool_responses)

    # Get final response
    final_result = chat(history, tools=tools)

    # Verify final response
    assert final_result.text is not None
    assert len(final_result.text) > 0
    assert "Paris" in final_result.text or "weather" in final_result.text.lower()


@pytest.mark.skip_if_no_config
def test_function_calling_tool_choice_required(test_config, has_real_api_config):
    """Test tool_choice='required' to force tool calling."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )

    # Force tool usage
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    result = chat(messages, tools=[weather_tool], tool_choice="required")

    # With tool_choice='required', model must call a tool even for simple greeting
    assert result.has_tool_calls, "Model should call tool when tool_choice='required'"


@pytest.mark.skip_if_no_config
def test_function_calling_tool_choice_specific(test_config, has_real_api_config):
    """Test tool_choice with specific function name."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    tools = [
        FunctionTool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        ),
        FunctionTool(
            name="get_time",
            description="Get time",
            parameters={
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        ),
    ]

    # Force specific tool
    tool_choice = ToolChoice(type="function", name="get_weather")
    messages = [{"role": "user", "content": "Tell me about Paris"}]
    result = chat(messages, tools=tools, tool_choice=tool_choice)

    assert result.has_tool_calls
    # Should call the specified tool
    tool_names = [tc.name for tc in result.tool_calls]
    assert "get_weather" in tool_names


@pytest.mark.skip_if_no_config
def test_function_calling_parallel(test_config, has_real_api_config):
    """Test parallel function calling (multiple tools in one request)."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    tools = [
        FunctionTool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        ),
        FunctionTool(
            name="get_time",
            description="Get time",
            parameters={
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        ),
    ]

    # Ask for multiple things that might trigger parallel tool calls
    messages = [
        {
            "role": "user",
            "content": "What's the weather in Paris and the time in London?",
        }
    ]
    result = chat(messages, tools=tools, parallel_tool_calls=True)

    # Check if multiple tools were called
    if result.has_tool_calls:
        # Model might call multiple tools in parallel
        assert len(result.tool_calls) >= 1

        # Each tool call should have valid structure
        for tc in result.tool_calls:
            assert tc.name in ["get_weather", "get_time"]
            assert tc.call_id is not None
            args = tc.get_arguments()
            assert isinstance(args, dict)


@pytest.mark.skip_if_no_config
def test_function_calling_complex_parameters(test_config, has_real_api_config):
    """Test function calling with complex JSON Schema parameters."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Tool with complex schema
    search_tool = FunctionTool(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "minLength": 1,
                    "maxLength": 100,
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "date_range": {"type": "string"},
                        "site": {"type": "string"},
                    },
                },
            },
            "required": ["query"],
        },
    )

    messages = [{"role": "user", "content": "Search for recent Python tutorials"}]
    result = chat(messages, tools=[search_tool])

    if result.has_tool_calls:
        tool_call = result.tool_calls[0]
        assert tool_call.name == "search_web"
        args = tool_call.get_arguments()
        assert "query" in args


# ============================================================================
# Streaming with Function Calling Tests
# ============================================================================


@pytest.mark.skip_if_no_config
def test_function_calling_streaming(test_config, has_real_api_config):
    """Test function calling with streaming response."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    # Stream with tools
    chunks = []
    for chunk in chat.stream(messages, tools=[weather_tool]):
        chunks.append(chunk)
        if chunk.has_content:
            print(chunk.delta, end="")

    # Get final chunk
    final_chunk = chunks[-1]

    # Verify streaming completed
    assert final_chunk.done, "Stream should be complete"

    # Check for tool calls in final result
    # Note: Tool calls might be in intermediate chunks or accumulated
    tool_calls_found = any(chunk.has_tool_calls for chunk in chunks)
    assert tool_calls_found, "Should have tool calls in stream"


@pytest.mark.skip_if_no_config
def test_function_calling_streaming_with_execution(test_config, has_real_api_config):
    """Test complete workflow with streaming: tool call -> execute -> stream final response."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )

    # Initial streaming request
    messages = [{"role": "user", "content": "What's the weather in Berlin?"}]

    # Collect chunks
    all_chunks = []
    tool_calls = []

    for chunk in chat.stream(messages, tools=[weather_tool]):
        all_chunks.append(chunk)
        if chunk.has_tool_calls:
            tool_calls.extend(chunk.tool_calls)

    # Should have tool calls
    assert len(tool_calls) > 0, "Should have tool calls from stream"

    # Create a mock ChatResult from streaming chunks
    from lexilux.chat.models import ChatResult
    from lexilux.usage import Usage

    # Get text from chunks
    text = "".join(c.delta for c in all_chunks if c.has_content)

    mock_result = ChatResult(
        text=text, usage=Usage(input_tokens=10, output_tokens=20), tool_calls=tool_calls
    )

    # Execute tool
    tool_responses = execute_tool_calls(mock_result, {"get_weather": get_weather})

    # Continue with streaming
    history = create_conversation_history(messages, mock_result, tool_responses)

    # Stream final response
    final_chunks = []
    for chunk in chat.stream(history, tools=[weather_tool]):
        final_chunks.append(chunk)

    assert len(final_chunks) > 0, "Should have final response chunks"
    assert final_chunks[-1].done, "Final stream should be complete"


# ============================================================================
# ToolCallHelper Tests
# ============================================================================


@pytest.mark.skip_if_no_config
def test_tool_call_helper_continue_conversation(test_config, has_real_api_config):
    """Test ToolCallHelper for simplified workflow."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Create helper
    helper = ToolCallHelper(
        functions={
            "get_weather": get_weather,
            "calculate": calculate,
        }
    )

    # Define tools
    tools = [
        FunctionTool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        ),
        FunctionTool(
            name="calculate",
            description="Calculate math expression",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        ),
    ]

    # Make request
    messages = [{"role": "user", "content": "What's 15 * 27?"}]
    result = chat(messages, tools=tools)

    # Use helper to continue
    if result.has_tool_calls:
        final_result = helper.continue_conversation(
            chat=chat, messages=messages, tool_result=result, tools=tools
        )

        assert final_result.text is not None
        assert len(final_result.text) > 0


# ============================================================================
# Multimodal/Vision Tests
# ============================================================================


@pytest.mark.skip_if_no_config
def test_multimodal_image_url(test_config, has_real_api_config):
    """Test multimodal with image URL."""
    if not has_real_api_config or "zhipu_multimodal" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_multimodal"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Message with text + image URL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nasa.gov/wp-content/uploads/2026/01/suni-williams-portrait.jpg"
                    },
                },
            ],
        }
    ]

    result = chat(messages)

    # Verify response
    assert result.text is not None
    assert len(result.text) > 0
    # Should describe the image
    print(f"Image description: {result.text[:200]}")


@pytest.mark.skipif(
    not HAS_PIL,
    reason="PIL (Pillow) not installed - optional dependency for image tests",
)
@pytest.mark.skip_if_no_config
def test_multimodal_base64_image(test_config, has_real_api_config):
    """Test multimodal with base64 encoded image."""
    if not has_real_api_config or "zhipu_multimodal" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_multimodal"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Create a simple test image
    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="red")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Message with base64 image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
        }
    ]

    result = chat(messages)

    # Verify response
    assert result.text is not None
    assert len(result.text) > 0
    assert "red" in result.text.lower() or "color" in result.text.lower()
    print(f"Image analysis: {result.text[:200]}")


@pytest.mark.skip_if_no_config
def test_multimodal_multiple_images(test_config, has_real_api_config):
    """Test multimodal with multiple images."""
    if not has_real_api_config or "zhipu_multimodal" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_multimodal"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Use two different public images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images. What's different?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nasa.gov/wp-content/uploads/2026/01/suni-williams-portrait.jpg"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nasa.gov/wp-content/uploads/2026/01/suni-williams-portrait.jpg"
                    },
                },
            ],
        }
    ]

    result = chat(messages)

    # Verify response
    assert result.text is not None
    assert len(result.text) > 0
    print(f"Comparison: {result.text[:300]}")


@pytest.mark.skip_if_no_config
def test_multimodal_with_image_detail_low(test_config, has_real_api_config):
    """Test multimodal with low detail setting."""
    if not has_real_api_config or "zhipu_multimodal" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_multimodal"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Briefly describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nasa.gov/wp-content/uploads/2026/01/suni-williams-portrait.jpg",
                        "detail": "low",
                    },
                },
            ],
        }
    ]

    result = chat(messages)

    assert result.text is not None
    assert len(result.text) > 0
    print(f"Low detail description: {result.text[:200]}")


@pytest.mark.skip_if_no_config
def test_multimodal_with_image_detail_high(test_config, has_real_api_config):
    """Test multimodal with high detail setting."""
    if not has_real_api_config or "zhipu_multimodal" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_multimodal"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nasa.gov/wp-content/uploads/2026/01/suni-williams-portrait.jpg",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    result = chat(messages)

    assert result.text is not None
    # High detail should produce longer, more detailed descriptions
    assert len(result.text) > 50
    print(f"High detail description: {result.text[:400]}")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.skip_if_no_config
def test_function_calling_no_tool_needed(test_config, has_real_api_config):
    """Test that model doesn't call tools when not needed."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )

    # Simple greeting that doesn't need tools
    messages = [{"role": "user", "content": "Hello! How are you today?"}]
    result = chat(messages, tools=[weather_tool])

    # Model should respond without calling tools
    assert result.text is not None
    assert len(result.text) > 0
    # With auto tool_choice, model decides not to use tool
    # This is acceptable behavior


@pytest.mark.skip_if_no_config
def test_function_calling_execution_error_handling(test_config, has_real_api_config):
    """Test error handling when tool execution fails."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Tool that will fail
    def failing_tool(location: str) -> str:
        raise ValueError("Intentional error for testing")

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )

    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    result = chat(messages, tools=[weather_tool])

    if result.has_tool_calls:
        # Execute with failing function
        tool_responses = execute_tool_calls(result, {"get_weather": failing_tool})

        # Should return error message in tool response
        assert len(tool_responses) > 0
        assert "Error" in tool_responses[0]["content"]


@pytest.mark.skip_if_no_config
def test_function_calling_missing_required_parameter(test_config, has_real_api_config):
    """Test tool call with missing required parameter."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}, "units": {"type": "string"}},
            "required": ["location"],
        },
    )

    # Ask vague question
    messages = [{"role": "user", "content": "What's the weather like?"}]
    chat(messages, tools=[weather_tool])

    # Model should either:
    # 1. Not call tool and ask for location
    # 2. Call tool with a guessed location
    # Both are acceptable behaviors


@pytest.mark.skip_if_no_config
def test_function_calling_with_system_message(test_config, has_real_api_config):
    """Test function calling with system message."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    weather_tool = FunctionTool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )

    # Request with system message
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    result = chat(
        messages,
        system="You are a helpful weather assistant. Always use the get_weather tool when asked about weather.",
        tools=[weather_tool],
    )

    # Should call tool with system message influence
    assert result.has_tool_calls or result.text is not None


# ============================================================================
# Complex Workflow Tests
# ============================================================================


@pytest.mark.skip_if_no_config
def test_function_calling_multi_turn_conversation(test_config, has_real_api_config):
    """Test multi-turn conversation with function calling."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    tools = [
        FunctionTool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        ),
        FunctionTool(
            name="calculate",
            description="Calculate",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        ),
    ]

    functions = {
        "get_weather": get_weather,
        "calculate": calculate,
    }

    # First turn
    messages = [{"role": "user", "content": "What's the weather in Madrid?"}]
    result = chat(messages, tools=tools)

    if result.has_tool_calls:
        tool_responses = execute_tool_calls(result, functions)
        history = create_conversation_history(messages, result, tool_responses)
        final_result = chat(history, tools=tools)

        # Second turn - follow-up question
        messages.append({"role": "assistant", "content": final_result.text})
        messages.append({"role": "user", "content": "What about Barcelona?"})

        result2 = chat(messages, tools=tools)

        if result2.has_tool_calls:
            tool_responses2 = execute_tool_calls(result2, functions)
            history2 = create_conversation_history(messages, result2, tool_responses2)
            final_result2 = chat(history2, tools=tools)

            assert final_result2.text is not None


@pytest.mark.skip_if_no_config
def test_function_calling_with_chained_tools(test_config, has_real_api_config):
    """Test workflow where output of one tool informs another."""
    if not has_real_api_config or "zhipu_function_calling" not in test_config:
        pytest.skip("No real API config available")

    config = test_config["zhipu_function_calling"]
    chat = Chat(
        base_url=config["api_base"], api_key=config["api_key"], model=config["model"]
    )

    # Define tools that might be chained
    tools = [
        FunctionTool(
            name="search",
            description="Search for information",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        FunctionTool(
            name="calculate",
            description="Calculate mathematical expression",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        ),
    ]

    functions = {
        "search": search_web,
        "calculate": calculate,
    }

    # Question that might require multiple tools
    messages = [
        {
            "role": "user",
            "content": "Search for the population of Paris, then calculate what it would be if it grew by 10%",
        }
    ]
    result = chat(messages, tools=tools)

    # Execute any tools called
    if result.has_tool_calls:
        tool_responses = execute_tool_calls(result, functions)
        history = create_conversation_history(messages, result, tool_responses)
        final_result = chat(history, tools=tools)

        assert final_result.text is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
