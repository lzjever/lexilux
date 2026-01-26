#!/usr/bin/env python
"""
41 Auto Continue - Handle Long Responses

Learn how to automatically continue responses that get cut off
due to max_tokens limits.

Level: Expert Topic
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat


def main():
    """Demonstrate auto-continue functionality."""
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

    # Example 1: Simple auto continue
    print("=" * 50)
    print("Example 1: Auto Continue Truncated Response")
    print("=" * 50)

    # Set a very low max_tokens to trigger truncation
    result = chat(
        "Write a detailed explanation of machine learning. "
        "Include at least 10 paragraphs.",
        max_tokens=50,  # Very small to force truncation
        auto_continue=True,
        max_continues=3,
    )

    print("Response with auto-continue:")
    print(result.text)
    print(f"\nFinish reason: {result.finish_reason}")
    print(f"Total tokens: {result.usage.total_tokens}")
    print(f"Continues used: {result.metadata.get('continue_count', 0)}\n")

    # Example 2: Manual continue
    print("=" * 50)
    print("Example 2: Manual Continue (more control)")
    print("=" * 50)

    # First request
    result = chat(
        "Explain quantum computing in detail",
        max_tokens=50,
    )

    print("First response:")
    print(result.text)
    print(f"Finish reason: {result.finish_reason}\n")

    # If truncated, continue
    if result.finish_reason == "length":
        print("Response was truncated. Continuing...\n")

        continued_result = chat.continue_chat(
            result,
            max_continues=2,
            max_tokens=50,
        )

        print("Continued response:")
        print(continued_result.text)
        print(f"\nFinal finish reason: {continued_result.finish_reason}")
        print(f"Total tokens: {continued_result.usage.total_tokens}\n")

    # Example 3: Streaming with continue
    print("=" * 50)
    print("Example 3: Streaming with Auto Continue")
    print("=" * 50)

    print("Streaming response with auto-continue:\n")

    for chunk in chat.stream(
        "Tell me everything about the history of computing",
        max_tokens=30,
        auto_continue=True,
        max_continues=2,
    ):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)

        if chunk.done:
            print(f"\n\nTotal tokens: {chunk.usage.total_tokens}")
            print(f"Finish reason: {chunk.finish_reason}")

            # Check if continues were used
            continue_count = chunk.metadata.get("continue_count", 0)
            if continue_count > 0:
                print(f"Continues used: {continue_count}")
    print()

    # Example 4: Custom continue prompt
    print("=" * 50)
    print("Example 4: Custom Continue Prompt")
    print("=" * 50)

    result = chat(
        "List 20 programming languages",
        max_tokens=30,
        auto_continue=True,
        continue_prompt="Continue the list from where you stopped",
        max_continues=3,
    )

    print("Response with custom continue prompt:")
    print(result.text)
    print(f"\nTokens: {result.usage.total_tokens}\n")

    # Example 5: Ensure complete response
    print("=" * 50)
    print("Example 5: Ensure Complete Response")
    print("=" * 50)

    print("Ensuring response completes (raises error if still truncated):\n")

    try:
        result = chat(
            "Write a very long essay about artificial intelligence",
            max_tokens=50,
            auto_continue=True,
            max_continues=2,
            ensure_complete=True,
        )
        print("Response completed successfully!")
        print(f"Tokens: {result.usage.total_tokens}")
    except Exception as e:
        print(f"Could not complete response: {e}")
        print("Try increasing max_continues or max_tokens")


if __name__ == "__main__":
    main()
