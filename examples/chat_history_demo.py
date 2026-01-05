#!/usr/bin/env python
"""
Chat History Demo

Demonstrates how to use ChatHistory for conversation management.
Shows automatic extraction, serialization, and multi-turn conversations.
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat
from lexilux.chat import ChatHistory


def get_chat_client(config_path=None):
    """Get Chat client with configuration"""
    args = parse_args()
    config_file = config_path or args.config

    try:
        config = get_chat_config(config_path=config_file)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading configuration: {e}")
        print("\nUsing default placeholder values. To use real API:")
        print("  1. Create tests/test_endpoints.json with your API credentials")
        print(
            "  2. Or specify a config file: python examples/chat_history_demo.py --config /path/to/config.json"
        )
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    return Chat(**config)


def demo_basic_history():
    """Demonstrate basic history management"""
    print("=" * 70)
    print("Demo 1: Basic Chat History")
    print("=" * 70)

    chat = get_chat_client()

    # Extract history from a Chat call
    print("\n1. Extract history from Chat call:")
    result = chat("What is Python?")
    history = ChatHistory.from_chat_result("What is Python?", result)
    print(f"   Messages: {len(history.messages)}")
    print(f"   System: {history.system}")

    # Continue the conversation
    print("\n2. Continue conversation:")
    result2 = chat(history.get_messages() + [{"role": "user", "content": "Tell me more"}])
    history = ChatHistory.from_chat_result(
        history.get_messages() + [{"role": "user", "content": "Tell me more"}],
        result2,
    )
    print(f"   Messages: {len(history.messages)}")
    print(f"   Complete conversation tracked automatically!")


def demo_serialization():
    """Demonstrate history serialization"""
    print("\n" + "=" * 70)
    print("Demo 2: History Serialization")
    print("=" * 70)

    chat = get_chat_client()

    # Create history
    result = chat("Explain recursion")
    history = ChatHistory.from_chat_result("Explain recursion", result)

    # Serialize to JSON
    print("\n1. Serialize to JSON:")
    json_str = history.to_json(indent=2)
    print(f"   JSON length: {len(json_str)} characters")
    print(f"   First 100 chars: {json_str[:100]}...")

    # Deserialize from JSON
    print("\n2. Deserialize from JSON:")
    history2 = ChatHistory.from_json(json_str)
    print(f"   Messages restored: {len(history2.messages)}")
    print(f"   System message: {history2.system}")


def demo_from_messages():
    """Demonstrate building history from messages"""
    print("\n" + "=" * 70)
    print("Demo 3: Building History from Messages")
    print("=" * 70)

    # From string
    print("\n1. From string:")
    history1 = ChatHistory.from_messages("Hello", system="You are helpful")
    print(f"   Messages: {len(history1.messages)}")
    print(f"   System: {history1.system}")

    # From list of strings
    print("\n2. From list of strings:")
    history2 = ChatHistory.from_messages(["Hello", "How are you?"])
    print(f"   Messages: {len(history2.messages)}")

    # From list of dicts
    print("\n3. From list of dicts:")
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]
    history3 = ChatHistory.from_messages(messages)
    print(f"   Messages: {len(history3.messages)}")
    print(f"   System: {history3.system}")


def main():
    """Main function to run all demos"""
    print("\n" + "=" * 70)
    print("Chat History Demo")
    print("=" * 70)
    print("\nThis demo shows how to use ChatHistory for conversation management.")
    print("Note: Replace 'your-api-key' with your actual API key to run.")
    print("\n" + "=" * 70)

    try:
        demo_basic_history()
        demo_serialization()
        demo_from_messages()

        print("\n" + "=" * 70)
        print("All demos completed!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: This demo requires a valid API key and endpoint.")
        print("Please update the API credentials in the code.")


if __name__ == "__main__":
    main()

