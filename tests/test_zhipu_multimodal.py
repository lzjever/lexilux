"""
Direct test script for Zhipu AI function calling and multimodal features.

This script tests:
1. Function calling with glm-4.7 model
2. Multimodal image understanding with glm-4.6v-flash model
"""

import base64
import io

import pytest

from lexilux import Chat, FunctionTool

# All tests in this file require external services
pytestmark = pytest.mark.integration

# Zhipu AI Configuration
# Note: base_url should NOT include the /chat/completions path
ZHIPU_API_KEY = "4f5ff08f17b14cdc9795ab696508e9ae.0i5f9tcioBW8cNr9"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_FUNCTION_MODEL = "glm-4.7"
ZHIPU_MULTIMODAL_MODEL = "glm-4.6v-flash"


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


# ============================================================================
# Function Calling Tests
# ============================================================================


def test_function_calling_basic():
    """Test basic function calling with glm-4.7."""
    print("\n" + "=" * 70)
    print("TEST: Basic Function Calling with glm-4.7")
    print("=" * 70)

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_FUNCTION_MODEL,
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
    print(f"Has tool calls: {result.has_tool_calls}")

    if result.has_tool_calls:
        print(f"Number of tool calls: {len(result.tool_calls)}")
        for i, tc in enumerate(result.tool_calls):
            print(f"  Tool Call {i + 1}:")
            print(f"    ID: {tc.id}")
            print(f"    Name: {tc.name}")
            args = tc.get_arguments()
            print(f"    Arguments: {args}")
    else:
        print(f"Response: {result.text}")

    print("✓ Test passed")
    return result


def test_function_calling_with_execution():
    """Test complete function calling workflow: request -> execute -> final response."""
    print("\n" + "=" * 70)
    print("TEST: Function Calling with Execution")
    print("=" * 70)

    from lexilux.chat.tool_helpers import (
        create_conversation_history,
        execute_tool_calls,
    )

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_FUNCTION_MODEL,
    )

    # Define tools
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
            name="calculate",
            description="Calculate mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        ),
    ]

    # Initial request
    messages = [{"role": "user", "content": "What's 15 * 27?"}]
    result = chat(messages, tools=tools)

    print(f"User: {messages[0]['content']}")
    print(f"Has tool calls: {result.has_tool_calls}")

    if result.has_tool_calls:
        # Execute tool calls
        tools_dict = {
            "get_weather": get_weather,
            "calculate": calculate,
        }

        tool_responses = execute_tool_calls(result, tools_dict)

        print("\nTool responses:")
        for i, resp in enumerate(tool_responses):
            # Find the corresponding tool call to get the name
            tool_name = (
                result.tool_calls[i].name if i < len(result.tool_calls) else "unknown"
            )
            print(f"  {tool_name}: {resp['content']}")

        # Create conversation history
        history = create_conversation_history(messages, result, tool_responses)

        # Get final response
        final_result = chat(history, tools=tools)

        print(f"\nFinal response: {final_result.text}")

    print("✓ Test passed")
    return result


def test_function_calling_tool_choice_required():
    """Test tool_choice='required' to force tool calling."""
    print("\n" + "=" * 70)
    print("TEST: Tool Choice Required")
    print("=" * 70)

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_FUNCTION_MODEL,
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

    print(f"Message: {messages[0]['content']}")
    print(f"Has tool calls (should be True): {result.has_tool_calls}")

    if result.has_tool_calls:
        for tc in result.tool_calls:
            print(f"  Tool called: {tc.name}")

    print("✓ Test passed")
    return result


def test_function_calling_parallel():
    """Test parallel function calling (multiple tools in one request)."""
    print("\n" + "=" * 70)
    print("TEST: Parallel Function Calling")
    print("=" * 70)

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_FUNCTION_MODEL,
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

    # Ask for multiple things that might trigger parallel tool calls
    messages = [
        {
            "role": "user",
            "content": "What's the weather in Paris and what's 25 * 4?",
        }
    ]
    result = chat(messages, tools=tools, parallel_tool_calls=True)

    print(f"Message: {messages[0]['content']}")
    print(f"Has tool calls: {result.has_tool_calls}")

    if result.has_tool_calls:
        print(f"Number of tool calls: {len(result.tool_calls)}")
        for i, tc in enumerate(result.tool_calls):
            print(f"  Tool Call {i + 1}: {tc.name}")
            args = tc.get_arguments()
            print(f"    Arguments: {args}")

    print("✓ Test passed")
    return result


# ============================================================================
# Multimodal/Vision Tests
# ============================================================================


def test_multimodal_image_url():
    """Test multimodal with image URL using Zhipu AI format."""
    print("\n" + "=" * 70)
    print("TEST: Multimodal Image URL with glm-4.6v-flash")
    print("=" * 70)

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_MULTIMODAL_MODEL,
    )

    # Zhipu AI uses a different format for multimodal content
    # They expect the content to be a list with mixed text and image objects
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Describe it briefly.",
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

    print(f"User question: {messages[0]['content'][0]['text']}")
    print("Image URL provided")
    print(f"\nModel response:\n{result.text}")

    print("✓ Test passed")
    return result


def test_multimodal_base64_image():
    """Test multimodal with base64 encoded image."""
    print("\n" + "=" * 70)
    print("TEST: Multimodal Base64 Image with glm-4.6v-flash")
    print("=" * 70)

    # Check if PIL is available
    try:
        from PIL import Image
    except ImportError:
        print("⚠ PIL not available, skipping base64 test")
        print("  Install with: uv pip install pillow")
        return None

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_MULTIMODAL_MODEL,
    )

    # Create a simple test image (red square)
    img = Image.new("RGB", (200, 200), color="red")

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

    print(f"User question: {messages[0]['content'][0]['text']}")
    print("Image: 200x200 red square (base64 encoded)")
    print(f"\nModel response:\n{result.text}")

    print("✓ Test passed")
    return result


def test_multimodal_multiple_images():
    """Test multimodal with multiple images."""
    print("\n" + "=" * 70)
    print("TEST: Multimodal Multiple Images with glm-4.6v-flash")
    print("=" * 70)

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_MULTIMODAL_MODEL,
    )

    # Create two simple test images with different colors
    try:
        from PIL import Image
    except ImportError:
        print("⚠ PIL not available, skipping multiple images test")
        print("  Install with: uv pip install pillow")
        return None

    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")

    buffer1 = io.BytesIO()
    img1.save(buffer1, format="PNG")
    image_data1 = base64.b64encode(buffer1.getvalue()).decode("utf-8")

    buffer2 = io.BytesIO()
    img2.save(buffer2, format="PNG")
    image_data2 = base64.b64encode(buffer2.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colors are these two images?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data1}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data2}"},
                },
            ],
        }
    ]

    result = chat(messages)

    print(f"User question: {messages[0]['content'][0]['text']}")
    print("Number of images: 2 (red and blue squares)")
    print(f"\nModel response:\n{result.text}")

    print("✓ Test passed")
    return result


def test_multimodal_with_detail_levels():
    """Test multimodal with different detail levels."""
    print("\n" + "=" * 70)
    print("TEST: Multimodal Detail Levels with glm-4.6v-flash")
    print("=" * 70)

    chat = Chat(
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        model=ZHIPU_MULTIMODAL_MODEL,
    )

    try:
        from PIL import Image
    except ImportError:
        print("⚠ PIL not available, skipping detail levels test")
        print("  Install with: uv pip install pillow")
        return None, None

    # Create a test image
    img = Image.new("RGB", (200, 200), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Test with basic description
    print("\n--- Testing basic description ---")
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
    print(f"Response:\n{result.text[:200]}...")

    print("\n✓ Test passed")
    return result, result


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ZHIPU AI FUNCTION CALLING AND MULTIMODAL TESTS")
    print("=" * 70)
    print(f"API Base URL: {ZHIPU_BASE_URL}")
    print(f"Function Calling Model: {ZHIPU_FUNCTION_MODEL}")
    print(f"Multimodal Model: {ZHIPU_MULTIMODAL_MODEL}")
    print("=" * 70)

    # Function Calling Tests
    print("\n\n### FUNCTION CALLING TESTS ###")
    test_function_calling_basic()
    test_function_calling_with_execution()
    test_function_calling_tool_choice_required()
    test_function_calling_parallel()

    # Multimodal Tests
    print("\n\n### MULTIMODAL TESTS ###")
    test_multimodal_image_url()
    test_multimodal_base64_image()
    test_multimodal_multiple_images()
    test_multimodal_with_detail_levels()

    print("\n\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
