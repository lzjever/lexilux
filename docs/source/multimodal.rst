Multimodal & Vision Support
============================

Lexilux provides comprehensive support for multimodal chat completions, enabling models to understand and analyze images alongside text.

Overview
--------

Multimodal chat allows models to:

* Process and analyze images from URLs or base64 encoding
* Support multiple images in a single request
* Combine text and visual content for rich interactions
* Control image detail levels for performance/cost optimization

Basic Multimodal Usage
----------------------

Image URL
~~~~~~~~~

Send an image URL for analysis:

.. code-block:: python

    from lexilux import Chat

    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-key",
        model="gpt-4-vision-preview"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            },
        ]
    }]

    result = chat(messages)
    print(result.text)

Base64 Encoded Images
~~~~~~~~~~~~~~~~~~~~~

Send base64 encoded images:

.. code-block:: python

    import base64
    from lexilux import Chat

    # Read and encode image
    with open("image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    chat = Chat(base_url="...", api_key="...", model="gpt-4-vision-preview")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            },
        ]
    }]

    result = chat(messages)
    print(result.text)

Multiple Images
~~~~~~~~~~~~~~~

Send multiple images in a single request:

.. code-block:: python

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two images"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image1.jpg"}
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image2.jpg"}
            },
        ]
    }]

    result = chat(messages)
    print(result.text)

Image Detail Levels
-------------------

Control image processing detail for cost/performance optimization:

.. code-block:: python

    from lexilux import Chat

    chat = Chat(base_url="...", api_key="...", model="gpt-4-vision-preview")

    # Low detail - faster, cheaper
    messages_low = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Briefly describe this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "low"  # 512x512, faster processing
                }
            },
        ]
    }]

    # High detail - more thorough analysis
    messages_high = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high"  # Full resolution, detailed analysis
                }
            },
        ]
    }]

Detail Levels
~~~~~~~~~~~~~

* ``"auto"`` - Let the model decide (default)
* ``"low"`` - Low resolution (512x512), faster and cheaper
* ``"high"`` - High resolution, more detailed analysis

Content Block Types
-------------------

Text Content Block
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    {"type": "text", "text": "Your question here"}

Image Content Block
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.jpg",
            "detail": "high"  # optional: "auto", "low", "high"
        }
    }

Type Aliases
~~~~~~~~~~~~

Lexilux provides type aliases for better IDE support:

* ``TextContentBlock`` - TypedDict for text content blocks
* ``ImageContentBlock`` - TypedDict for image content blocks
* ``ImageUrlDetail`` - TypedDict for image URL with optional detail
* ``ContentBlock`` - Union of all content block types

.. code-block:: python

    from lexilux import TextContentBlock, ImageContentBlock, ContentBlock

    text_block: TextContentBlock = {"type": "text", "text": "Hello"}
    image_block: ImageContentBlock = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"}
    }

Multimodal with Function Calling
---------------------------------

Combine image analysis with function calling:

.. code-block:: python

    from lexilux import Chat, FunctionTool

    chat = Chat(base_url="...", api_key="...", model="gpt-4-vision-preview")

    # Define tool
    extract_color_tool = FunctionTool(
        name="extract_dominant_color",
        description="Extract dominant color from image",
        parameters={
            "type": "object",
            "properties": {
                "color_hex": {"type": "string", "description": "Hex color code"}
            },
            "required": ["color_hex"]
        }
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the dominant color in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            },
        ]
    }]

    result = chat(messages, tools=[extract_color_tool])

    if result.has_tool_calls:
        for tool_call in result.tool_calls:
            print(f"Tool called: {tool_call.name}")
            print(f"Arguments: {tool_call.get_arguments()}")

Streaming with Multimodal
--------------------------

Multimodal works seamlessly with streaming:

.. code-block:: python

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            },
        ]
    }]

    for chunk in chat.stream(messages):
        if chunk.has_content:
            print(chunk.delta, end="")

Best Practices
--------------

1. **Use Appropriate Detail Levels**: Choose detail level based on your needs - use "low" for quick analysis and "high" for detailed understanding.

2. **Optimize Image Sizes**: Resize large images before sending to reduce token usage and improve response times.

3. **Use Base64 for Privacy**: For sensitive images, use base64 encoding instead of public URLs.

4. **Handle Mixed Content**: Always structure content as a list of blocks when combining text and images.

5. **Test with Different Models**: Not all models support multimodal - check your provider's documentation.

6. **Consider Rate Limits**: Multimodal requests may have different rate limits than text-only requests.

Common Use Cases
----------------

Document Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this document image"},
            {"type": "image_url", "image_url": {"url": document_image_url}}
        ]
    }]

Image Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Classify this image into one of: cat, dog, bird, other"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]

Visual Question Answering
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "How many people are in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]

Data Extraction
~~~~~~~~~~~~~~~

.. code-block:: python

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract the receipt total and date from this image"},
            {"type": "image_url", "image_url": {"url": receipt_image_url}}
        ]
    }]

API Reference
-------------

For complete API documentation, see:

* :class:`~lexilux.chat.content_blocks.TextContentBlock` - Text content block
* :class:`~lexilux.chat.content_blocks.ImageContentBlock` - Image content block
* :class:`~lexilux.chat.content_blocks.ImageUrlDetail` - Image URL with detail
* :class:`~lexilux.chat.Chat` - Chat client with multimodal support

Provider Compatibility
----------------------

Multimodal support depends on the model and provider:

* **OpenAI**: GPT-4V, GPT-4o - Full multimodal support
* **Anthropic**: Claude 3.5 Sonnet - Full multimodal support
* **Google**: Gemini Pro Vision - Full multimodal support
* **Zhipu AI**: glm-4v, glm-4.6v-flash - Full multimodal support

Always check your provider's documentation for specific multimodal capabilities and limitations.
