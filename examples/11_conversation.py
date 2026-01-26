#!/usr/bin/env python
"""
11 Multi-turn Conversation - Chat That Remembers

Learn how to have multi-turn conversations where the AI remembers context.
You can pass message history to maintain conversation state.

Level: Core Feature
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat


def main():
    """Demonstrate multi-turn conversations."""
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

    # Example 1: Manual conversation history
    print("=" * 50)
    print("Example 1: Manual Conversation History")
    print("=" * 50)

    # Start with a user message
    messages = [{"role": "user", "content": "My favorite color is blue."}]
    result = chat(messages)
    print(f"AI: {result.text}")

    # Add AI response to history and ask a follow-up
    messages.append({"role": "assistant", "content": result.text})
    messages.append({"role": "user", "content": "What's my favorite color?"})
    result = chat(messages)
    print(f"AI: {result.text}\n")

    # Example 2: Building a simple chat loop
    print("=" * 50)
    print("Example 2: Simple Chat Loop")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    conversation_history = []
    system_message = "You are a friendly and helpful assistant."

    while True:
        user_input = input("You: ")

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Build messages with history
        messages = [{"role": "system", "content": system_message}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})

        # Get response
        result = chat(messages)
        print(f"AI: {result.text}")

        # Add to history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": result.text})

        # Keep history manageable (last 10 messages)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]


if __name__ == "__main__":
    main()
