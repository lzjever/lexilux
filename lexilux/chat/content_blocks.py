"""
Content block types for multimodal messages.

This module defines TypedDict types for different content blocks in messages,
supporting text, images, and tool calls in a unified structure.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, Union


class TextContentBlock(TypedDict):
    """
    Text content block.

    Represents a plain text content in a message.

    Attributes:
        type: Content type identifier, always "text".
        text: The text content string.

    Examples:
        >>> block: TextContentBlock = {"type": "text", "text": "Hello, world!"}
    """
    type: Literal["text"]
    text: str


class ImageUrlDetail(TypedDict):
    """
    Image URL details with optional resolution settings.

    Attributes:
        url: Image URL (can be HTTP/HTTPS URL or base64 data URL).
        detail: Optional detail level for image processing.
            - "auto": Let model decide (default).
            - "low": Low resolution (512x512), faster and cheaper.
            - "high": High resolution, more detailed analysis.

    Examples:
        >>> url_detail: ImageUrlDetail = {
        ...     "url": "https://example.com/image.jpg"
        ... }
        >>> url_detail_with_detail: ImageUrlDetail = {
        ...     "url": "https://example.com/image.jpg",
        ...     "detail": "high"
        ... }
    """
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ImageContentBlock(TypedDict):
    """
    Image content block.

    Represents an image in a message, used for vision/multimodal capabilities.

    Attributes:
        type: Content type identifier, always "image_url".
        image_url: Image URL details including URL and optional detail level.

    Examples:
        >>> block: ImageContentBlock = {
        ...     "type": "image_url",
        ...     "image_url": {"url": "https://example.com/image.jpg"}
        ... }
        >>> # With base64 encoding
        >>> block_base64: ImageContentBlock = {
        ...     "type": "image_url",
        ...     "image_url": {
        ...         "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        ...     }
        ... }
    """
    type: Literal["image_url"]
    image_url: ImageUrlDetail


class ToolCallBlock(TypedDict):
    """
    Tool call content block (for assistant responses only).

    Represents a tool call initiated by the model in a response message.

    Attributes:
        type: Content type identifier, always "tool_call".
        id: Unique identifier for this tool call.
        call_id: Call reference ID (used for submitting tool outputs).
        name: Name of the function/tool being called.
        arguments: JSON string containing function arguments.

    Examples:
        >>> block: ToolCallBlock = {
        ...     "type": "tool_call",
        ...     "id": "call_abc123",
        ...     "call_id": "call_abc123",
        ...     "name": "get_weather",
        ...     "arguments": '{"location": "Paris", "units": "celsius"}'
        ... }
    """
    type: Literal["tool_call"]
    id: str
    call_id: str
    name: str
    arguments: str


# Union type for all content block types
ContentBlock = Union[TextContentBlock, ImageContentBlock, ToolCallBlock]

# Type alias for content in messages (string or list of blocks)
MessageContent = Union[str, list[ContentBlock]]
