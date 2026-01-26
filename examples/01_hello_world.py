#!/usr/bin/env python
"""
01 Hello World - Simplest Possible Chat Call

This is the absolute simplest way to use Lexilux.
Perfect for your first chat completion!

Level: Beginner
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat


def main():
    """Run the simplest possible chat example."""
    # Load configuration
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

    # Create a chat client
    chat = Chat(**config)

    # Make a simple chat call
    result = chat("Hello, world!")

    # Print the response
    print("=" * 50)
    print("Response:", result.text)
    print("=" * 50)
    print("Tokens used:", result.usage.total_tokens)
    print("Finish reason:", result.finish_reason)


if __name__ == "__main__":
    main()
