Function Calling Guide
======================

Lexilux provides comprehensive support for OpenAI-compatible function calling (also known as tool use),
allowing models to request execution of specific functions during chat completions.

Overview
--------

Function calling enables models to:

* Request execution of specific functions with structured arguments
* Process function results and generate contextual responses
* Support multiple tools in a single request (parallel function calling)
* Combine function calls with regular text generation

Basic Function Calling
----------------------

Defining Tools
~~~~~~~~~~~~~~

Define tools using :class:`~lexilux.chat.tools.FunctionTool` with JSON Schema parameters:

.. code-block:: python

    from lexilux import FunctionTool

    get_weather_tool = FunctionTool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. Paris"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    )

Making Requests with Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass tools to the chat client:

.. code-block:: python

    from lexilux import Chat

    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-key",
        model="gpt-4"
    )

    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    result = chat(messages, tools=[get_weather_tool])

    # Check if model requested tool calls
    if result.has_tool_calls:
        for tool_call in result.tool_calls:
            print(f"Function: {tool_call.name}")
            print(f"Arguments: {tool_call.get_arguments()}")

Executing Tool Calls
~~~~~~~~~~~~~~~~~~~~~

Use helper functions to execute tools and continue the conversation:

.. code-block:: python

    from lexilux import execute_tool_calls, create_conversation_history

    # Define your actual function implementations
    def get_weather(location: str, units: str = "celsius") -> str:
        # Your implementation here
        return f"Weather in {location}: 22°{units}"

    # Execute tool calls
    tool_responses = execute_tool_calls(
        result,
        {"get_weather": get_weather}
    )

    # Create conversation history
    history = create_conversation_history(messages, result, tool_responses)

    # Get final response
    final_result = chat(history, tools=[get_weather_tool])
    print(final_result.text)

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example showing the full workflow:

.. code-block:: python

    from lexilux import Chat, FunctionTool, execute_tool_calls, create_conversation_history

    # 1. Define tools
    tools = [
        FunctionTool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        ),
        FunctionTool(
            name="get_time",
            description="Get current time for a timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": ["timezone"]
            }
        )
    ]

    # 2. Define function implementations
    def get_weather(location: str, units: str = "celsius") -> str:
        return f"Weather in {location}: 22°{units}"

    def get_time(timezone: str) -> str:
        from datetime import datetime
        # Implementation here
        return f"Time in {timezone}: {datetime.now().isoformat()}"

    functions = {
        "get_weather": get_weather,
        "get_time": get_time,
    }

    # 3. Make request
    chat = Chat(base_url="...", api_key="...", model="gpt-4")
    messages = [{"role": "user", "content": "What's the weather and time in Paris?"}]
    result = chat(messages, tools=tools)

    # 4. Execute and continue
    if result.has_tool_calls:
        tool_responses = execute_tool_calls(result, functions)
        history = create_conversation_history(messages, result, tool_responses)
        final_result = chat(history, tools=tools)
        print(final_result.text)

Controlling Tool Usage
----------------------

Tool Choice
~~~~~~~~~~~

Control when and how the model uses tools with the ``tool_choice`` parameter:

.. code-block:: python

    from lexilux import ToolChoice

    # Auto mode (default) - model decides
    result = chat(messages, tools=[tool], tool_choice="auto")

    # Required mode - model must call at least one tool
    result = chat(messages, tools=[tool], tool_choice="required")

    # Specific function - model must call this function
    result = chat(messages, tools=[tool], tool_choice=ToolChoice(
        type="function",
        name="get_weather"
    ))

Parallel Tool Calls
~~~~~~~~~~~~~~~~~~~

Enable parallel function calling (multiple tools in one request):

.. code-block:: python

    result = chat(
        messages,
        tools=[tool1, tool2, tool3],
        parallel_tool_calls=True
    )

    if result.has_tool_calls:
        # Model may have called multiple tools
        for tool_call in result.tool_calls:
            print(f"Called: {tool_call.name}")

Using ToolCallHelper
--------------------

The :class:`~lexilux.chat.tool_helpers.ToolCallHelper` class provides a high-level abstraction
for managing tool calling workflows:

.. code-block:: python

    from lexilux import ToolCallHelper

    # Create helper with function implementations
    helper = ToolCallHelper(functions={
        "get_weather": get_weather,
        "calculate": calculate,
    })

    # Make initial request
    result = chat("What's 15 * 27?", tools=[calculate_tool])

    # Automatically execute and continue
    if result.has_tool_calls:
        final_result = helper.continue_conversation(
            chat=chat,
            messages=[{"role": "user", "content": "What's 15 * 27?"}],
            tool_result=result,
            tools=[calculate_tool]
        )
        print(final_result.text)

Streaming with Function Calling
-------------------------------

Function calling works seamlessly with streaming:

.. code-block:: python

    # Stream initial request
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    chunks = []
    tool_calls = []

    for chunk in chat.stream(messages, tools=[weather_tool]):
        chunks.append(chunk)
        if chunk.has_tool_calls:
            tool_calls.extend(chunk.tool_calls)

    # Execute and stream final response
    if tool_calls:
        tool_responses = execute_tool_calls(
            # Create ChatResult from chunks
            result,
            functions
        )
        history = create_conversation_history(messages, result, tool_responses)

        # Stream final response
        for chunk in chat.stream(history, tools=[weather_tool]):
            print(chunk.delta, end="")

Error Handling
--------------

Handling Unknown Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~lexilux.chat.tool_helpers.execute_tool_calls` function raises ``ValueError``
for unknown functions:

.. code-block:: python

    try:
        tool_responses = execute_tool_calls(result, functions)
    except ValueError as e:
        print(f"Unknown function requested: {e}")

Handling Execution Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

If a function raises an exception during execution, the error message is returned
to the model in the tool response:

.. code-block:: python

    def failing_tool(location: str) -> str:
        raise ValueError("Intentional error")

    # Error message will be: "Error executing failing_tool: Intentional error"
    tool_responses = execute_tool_calls(result, {"failing_tool": failing_tool})

Best Practices
--------------

1. **Define Clear Descriptions**: Provide detailed descriptions for tools to help the model understand when to use them.

2. **Use JSON Schema Validation**: Define proper JSON Schema with types, required fields, and enums for better argument validation.

3. **Handle Errors Gracefully**: Always handle exceptions in your function implementations and return meaningful error messages.

4. **Check Tool Calls**: Always check ``result.has_tool_calls`` before attempting to access ``result.tool_calls``.

5. **Use Type Hints**: Add type hints to your function implementations for better clarity and IDE support.

6. **Test Functions**: Test your function implementations independently before integrating with the chat API.

Advanced Topics
---------------

Strict Schema Mode
~~~~~~~~~~~~~~~~~~

Enable strict schema validation for tool parameters:

.. code-block:: python

    tool = FunctionTool(
        name="search",
        description="Search database",
        parameters={...},
        strict=True  # Enforce strict schema adherence
    )

Tool Response Format
~~~~~~~~~~~~~~~~~~~~

Tool responses are formatted as messages with ``role="tool"`` and include:

* ``tool_call_id`` - ID of the tool call being responded to
* ``content`` - String result from the function execution

Multi-Turn Conversations
~~~~~~~~~~~~~~~~~~~~~~~~~

For complex workflows requiring multiple tool calls:

.. code-block:: python

    history = []

    # Turn 1
    result1 = chat(messages, tools=tools)
    if result1.has_tool_calls:
        responses1 = execute_tool_calls(result1, functions)
        history = create_conversation_history(messages, result1, responses1)
        result1 = chat(history, tools=tools)

    # Turn 2 - follow-up question
    messages.append({"role": "assistant", "content": result1.text})
    messages.append({"role": "user", "content": "What about London?"})
    result2 = chat(messages, tools=tools)

    # Continue workflow...

API Reference
-------------

For complete API documentation, see:

* :class:`~lexilux.chat.tools.FunctionTool` - Tool definition
* :class:`~lexilux.chat.tools.ToolChoice` - Tool choice configuration
* :class:`~lexilux.chat.models.ToolCall` - Tool call in response
* :class:`~lexilux.chat.tool_helpers.ToolCallHelper` - High-level workflow helper
* :func:`~lexilux.chat.tool_helpers.execute_tool_calls` - Execute tool calls
* :func:`~lexilux.chat.tool_helpers.create_conversation_history` - Build conversation history
