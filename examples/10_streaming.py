#!/usr/bin/env python
"""
10 Streaming - Real-time Response Generation

Learn how to receive responses in real-time as they're generated.
Streaming makes your application feel faster and more responsive.

Level: Core Feature
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat


def main():
    """Demonstrate streaming chat completions."""
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

    # Example 1: Basic streaming
    print("=" * 50)
    print("Example 1: Basic Streaming")
    print("=" * 50)
    print("Response: ", end="", flush=True)

    for chunk in chat.stream("Tell me a short joke about programming"):
        print(chunk.delta, end="", flush=True)
        if chunk.done:
            print(f"\n\nTokens used: {chunk.usage.total_tokens}")
    print()

    # Example 2: Streaming with more control
    print("=" * 50)
    print("Example 2: Streaming with Full Control")
    print("=" * 50)

    full_response = ""
    for chunk in chat.stream("Count from 1 to 5"):
        if chunk.delta:
            full_response += chunk.delta
            print(chunk.delta, end="", flush=True)

        # Check if we're done
        if chunk.done:
            print(f"\n\nComplete response: {full_response}")
            print(f"Input tokens: {chunk.usage.input_tokens}")
            print(f"Output tokens: {chunk.usage.output_tokens}")
            print(f"Total tokens: {chunk.usage.total_tokens}")
            print(f"Finish reason: {chunk.finish_reason}")
    print()

    # Example 3: Streaming for long content
    print("=" * 50)
    print("Example 3: Streaming Long Content")
    print("=" * 50)
    print("Ask a question that requires a longer answer...\n")

    question = "What are the main differences between Python and JavaScript?"
    for chunk in chat.stream(question):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        if chunk.done:
            print(f"\n\n(Total tokens: {chunk.usage.total_tokens})")


if __name__ == "__main__":
    main()
