#!/usr/bin/env python
"""
12 Chat Parameters - Control Response Behavior

Learn how to use parameters to control the AI's responses:
- temperature: Controls randomness/creativity (0.0 to 2.0)
- max_tokens: Limits response length
- stop: Stops generation at specified sequences

Level: Core Feature
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat, ChatParams


def main():
    """Demonstrate chat parameter usage."""
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

    # Example 1: Temperature (creativity control)
    print("=" * 50)
    print("Example 1: Temperature Comparison")
    print("=" * 50)

    prompt = "Write a short creative opening for a sci-fi story."

    print("\nTemperature = 0.0 (focused, deterministic):")
    result = chat(prompt, temperature=0.0)
    print(f"{result.text}\n")

    print("Temperature = 1.0 (balanced):")
    result = chat(prompt, temperature=1.0)
    print(f"{result.text}\n")

    print("Temperature = 1.8 (creative, random):")
    result = chat(prompt, temperature=1.8)
    print(f"{result.text}\n")

    # Example 2: Max tokens (length control)
    print("=" * 50)
    print("Example 2: Max Tokens Control")
    print("=" * 50)

    print("Max tokens = 20 (very short):")
    result = chat("What is Python?", max_tokens=20)
    print(f"{result.text}")
    print(f"Tokens used: {result.usage.output_tokens}")
    print(f"Finish reason: {result.finish_reason}\n")

    print("Max tokens = 100 (longer):")
    result = chat("What is Python?", max_tokens=100)
    print(f"{result.text}")
    print(f"Tokens used: {result.usage.output_tokens}")
    print(f"Finish reason: {result.finish_reason}\n")

    # Example 3: Stop sequences
    print("=" * 50)
    print("Example 3: Stop Sequences")
    print("=" * 50)

    print("Stop at '---' or 'END':")
    result = chat(
        "List 5 programming languages. Separate each with '---'",
        stop=["---", "END"],
    )
    print(f"{result.text}")
    print(f"Finish reason: {result.finish_reason}\n")

    # Example 4: Using ChatParams for reusable configuration
    print("=" * 50)
    print("Example 4: ChatParams (Reusable Configuration)")
    print("=" * 50)

    # Create a params object for consistent behavior
    creative_params = ChatParams(
        temperature=1.5,
        max_tokens=50,
        stop=["\n\n"],  # Stop at double newline
    )

    print("Using ChatParams for creative, short responses:")
    for i in range(3):
        result = chat("Tell me a fun fact", params=creative_params)
        print(f"{i + 1}. {result.text.strip()}")


if __name__ == "__main__":
    main()
