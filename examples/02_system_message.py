#!/usr/bin/env python
"""
02 System Message - Setting Up Chat Behavior

Learn how to use system messages to control how the AI responds.
System messages set the behavior, personality, or rules for the AI.

Level: Beginner
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat


def main():
    """Demonstrate system message usage."""
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

    # Example 1: Without system message
    print("=" * 50)
    print("Without system message:")
    print("=" * 50)
    result = chat("What is Python?")
    print(result.text)
    print()

    # Example 2: With helpful assistant system message
    print("=" * 50)
    print("With system message (helpful assistant):")
    print("=" * 50)
    result = chat(
        "What is Python?",
        system="You are a helpful assistant that explains "
        "technical concepts clearly and concisely.",
    )
    print(result.text)
    print()

    # Example 3: With code expert system message
    print("=" * 50)
    print("With system message (code expert):")
    print("=" * 50)
    result = chat(
        "What is Python?",
        system="You are a senior software engineer. "
        "Answer with code examples and best practices.",
    )
    print(result.text)
    print()

    # Example 4: With constraint system message
    print("=" * 50)
    print("With system message (short answers only):")
    print("=" * 50)
    result = chat(
        "What is Python?",
        system="Answer in ONE sentence only. Be concise.",
    )
    print(result.text)


if __name__ == "__main__":
    main()
