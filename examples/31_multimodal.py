#!/usr/bin/env python
"""
31 Multimodal - Text and Images Together

Learn how to process images along with text.
This enables vision capabilities like analyzing images, reading charts, etc.

Level: Advanced Feature
"""

import base64
from pathlib import Path

from config_loader import get_chat_config, parse_args

from lexilux import Chat, ImageContentBlock, TextContentBlock


def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    """Demonstrate multimodal capabilities."""
    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4-vision-preview",
        }

    chat = Chat(**config)

    # Example 1: Image from URL
    print("=" * 50)
    print("Example 1: Analyze Image from URL")
    print("=" * 50)

    messages = [
        {
            "role": "user",
            "content": [
                TextContentBlock(text="What's in this image? Describe it briefly."),
                ImageContentBlock(
                    image_url={
                        "url": "https://raw.githubusercontent.com/python/python-logo/main/Python-logo-notext.svg"
                    }
                ),
            ],
        }
    ]

    try:
        result = chat(messages)
        print(f"Response: {result.text}\n")
    except Exception as e:
        print("Note: This example requires a vision-enabled model.")
        print(f"Error: {e}\n")

    # Example 2: Local image (base64 encoded)
    print("=" * 50)
    print("Example 2: Analyze Local Image")
    print("=" * 50)

    # Check if we have a sample image
    sample_images = [
        "examples/sample_image.png",
        "examples/sample_image.jpg",
        "/tmp/sample_image.png",
    ]

    image_path = None
    for path in sample_images:
        if Path(path).exists():
            image_path = path
            break

    if image_path:
        base64_image = encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    TextContentBlock(text="Describe this image in detail."),
                    ImageContentBlock(
                        image_url={"url": f"data:image/png;base64,{base64_image}"}
                    ),
                ],
            }
        ]

        try:
            result = chat(messages)
            print(f"Response: {result.text}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    else:
        print("No sample image found. To test local images:")
        print("  1. Place an image at examples/sample_image.png")
        print("  2. Or update the image_path variable\n")

    # Example 3: Multiple images
    print("=" * 50)
    print("Example 3: Compare Multiple Images")
    print("=" * 50)

    messages = [
        {
            "role": "user",
            "content": [
                TextContentBlock(
                    text="What are the differences between these two images?"
                ),
                ImageContentBlock(
                    image_url={"url": "https://placehold.co/100x100/red/red"}
                ),
                ImageContentBlock(
                    image_url={"url": "https://placehold.co/100x100/blue/blue"}
                ),
            ],
        }
    ]

    try:
        result = chat(messages)
        print(f"Response: {result.text}\n")
    except Exception as e:
        print("Note: This example requires network access and vision model.")
        print(f"Error: {e}\n")

    # Example 4: Image detail levels
    print("=" * 50)
    print("Example 4: Image Detail Levels")
    print("=" * 50)

    print("Different detail levels for different use cases:\n")

    print("Low detail - faster, less detailed analysis:")
    print("  ImageUrlDetail.LOW\n")

    print("High detail - slower, more detailed analysis:")
    print("  ImageUrlDetail.HIGH\n")

    print("Auto detail - lets the model decide:")
    print("  ImageUrlDetail.AUTO\n")

    # Example usage
    from lexilux import ImageUrlDetail

    messages = [
        {
            "role": "user",
            "content": [
                TextContentBlock(text="Quick: what color is this?"),
                ImageContentBlock(
                    image_url={
                        "url": "https://placehold.co/100x100/green/green",
                        "detail": ImageUrlDetail.LOW,
                    }
                ),
            ],
        }
    ]

    try:
        result = chat(messages)
        print(f"\nWith LOW detail: {result.text}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 5: Reading text from images (OCR)
    print("=" * 50)
    print("Example 5: OCR - Reading Text from Images")
    print("=" * 50)

    messages = [
        {
            "role": "user",
            "content": [
                TextContentBlock(text="Read and transcribe any text in this image."),
                ImageContentBlock(
                    image_url={
                        "url": "https://placehold.co/300x100/000000/FFF?text=Hello+World"
                    }
                ),
            ],
        }
    ]

    try:
        result = chat(messages)
        print(f"Transcribed text: {result.text}\n")
    except Exception as e:
        print("Note: This requires a vision-enabled model.")
        print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
