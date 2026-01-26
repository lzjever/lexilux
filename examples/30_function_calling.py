#!/usr/bin/env python
"""
30 Function Calling - Let AI Call Your Functions

Learn how to give the AI access to tools/functions.
The AI can decide when and how to call your functions.

Level: Advanced Feature
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat, FunctionTool, create_conversation_history, execute_tool_calls


def main():
    """Demonstrate function calling."""
    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    chat = Chat(**config)

    # Example 1: Simple function calling
    print("=" * 50)
    print("Example 1: Simple Function Calling")
    print("=" * 50)

    # Define a function the AI can call
    get_weather = FunctionTool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. Paris, Tokyo",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )

    # Make a request with the tool
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]

    result = chat(messages, tools=[get_weather])

    # Check if AI wants to call a function
    if result.has_tool_calls:
        print("AI wants to call a function:")
        for tool_call in result.tool_calls:
            print(f"  Function: {tool_call.name}")
            print(f"  Arguments: {tool_call.get_arguments()}")
    else:
        print(f"AI response: {result.text}")

    # Example 2: Execute the function and continue
    print("\n" + "=" * 50)
    print("Example 2: Execute and Continue Conversation")
    print("=" * 50)

    # Simulated function implementations
    def get_weather_impl(location: str, unit: str = "celsius") -> str:
        """Simulated weather function."""
        temps = {
            "Paris": "22°C",
            "Tokyo": "28°C",
            "New York": "25°C",
        }
        temp = temps.get(location, "20°C")
        return f"The weather in {location} is {temp}."

    def get_time_impl(location: str) -> str:
        """Simulated time function."""
        times = {
            "Paris": "14:30",
            "Tokyo": "21:30",
            "New York": "08:30",
        }
        time = times.get(location, "12:00")
        return f"The current time in {location} is {time}."

    # Define another tool
    get_time = FunctionTool(
        name="get_time",
        description="Get the current time for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": ["location"],
        },
    )

    tools = [get_weather, get_time]
    tool_map = {
        "get_weather": get_weather_impl,
        "get_time": get_time_impl,
    }

    # Make a request
    messages = [{"role": "user", "content": "What's the weather and time in Paris?"}]

    result = chat(messages, tools=tools)

    # Loop until AI has all information it needs
    max_iterations = 5
    for i in range(max_iterations):
        if result.has_tool_calls:
            print(f"\nIteration {i + 1}: Executing tool calls...")

            # Execute the tool calls
            tool_responses = execute_tool_calls(result, tool_map)

            for response in tool_responses:
                print(f"  Tool result: {response.content}")

            # Continue conversation with tool results
            messages = create_conversation_history(messages, result, tool_responses)
            result = chat(messages, tools=tools)
        else:
            print(f"\nFinal answer: {result.text}")
            break

    # Example 3: Parallel function calls
    print("\n" + "=" * 50)
    print("Example 3: Parallel Function Calls")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "What's the weather in Paris, Tokyo, and New York?"}
    ]

    result = chat(messages, tools=[get_weather])

    if result.has_tool_calls:
        print(f"AI wants to call {len(result.tool_calls)} functions in parallel:")
        for tool_call in result.tool_calls:
            print(f"  - {tool_call.name}({tool_call.get_arguments()})")


if __name__ == "__main__":
    main()
