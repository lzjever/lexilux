#!/usr/bin/env python
"""
40 Chat History - Advanced Conversation Management

Learn how to use ChatHistory for sophisticated conversation tracking.
This includes history formatting, searching, and merging.

Level: Expert Topic
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat, ChatHistory, ChatHistoryFormatter, Conversation


def main():
    """Demonstrate ChatHistory features."""
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

    # Example 1: Building conversation history
    print("=" * 50)
    print("Example 1: Building Conversation History")
    print("=" * 50)

    # Create a conversation
    conv = Conversation()

    # Add messages
    result = conv.chat(chat, "Hello, I'm learning Python.")
    print("You: Hello, I'm learning Python.")
    print(f"AI: {result.text}\n")

    result = conv.chat(chat, "What are the key features?")
    print("You: What are the key features?")
    print(f"AI: {result.text}\n")

    # Get the history
    history = conv.to_history()
    print(f"History has {len(history)} messages")
    print(f"Total tokens: {history.usage.total_tokens}\n")

    # Example 2: ChatHistory from messages
    print("=" * 50)
    print("Example 2: Creating ChatHistory")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Why is it popular?"},
    ]

    history = ChatHistory.from_messages(messages)

    print("ChatHistory created from messages:")
    print(f"  Messages: {len(history.messages)}")
    print(f"  Total tokens: {history.usage.total_tokens}\n")

    # Example 3: Formatting history
    print("=" * 50)
    print("Example 3: Formatting Conversation")
    print("=" * 50)

    # Get a longer conversation
    conv = Conversation()
    conv.chat(chat, "Tell me about Python lists")
    conv.chat(chat, "How do I append to a list?")
    conv.chat(chat, "What about list comprehensions?")

    history = conv.to_history()

    # Format as markdown
    formatter = ChatHistoryFormatter()
    markdown = formatter.format_markdown(history)

    print("Markdown format:")
    print(markdown)
    print()

    # Example 4: Saving and loading history
    print("=" * 50)
    print("Example 4: Save and Load History")
    print("=" * 50)

    # Save to JSON
    json_data = history.to_json()
    print(f"History saved as JSON ({len(json_data)} characters)\n")

    # Load from JSON
    loaded_history = ChatHistory.from_json(json_data)
    print(f"History loaded: {len(loaded_history.messages)} messages\n")

    # Example 5: Searching history
    print("=" * 50)
    print("Example 5: Search in History")
    print("=" * 50)

    from lexilux import search_content

    # Search for keywords
    results = search_content(history, "list")

    print(f"Messages containing 'list': {len(results)}")
    for msg in results:
        print(f"  [{msg.role}]: {msg.content[:60]}...")
    print()

    # Example 6: Filter by role
    print("=" * 50)
    print("Example 6: Filter by Role")
    print("=" * 50)

    from lexilux import filter_by_role

    user_messages = filter_by_role(history, "user")
    print(f"User messages: {len(user_messages)}")
    for msg in user_messages:
        print(f"  - {msg.content[:60]}...")
    print()

    # Example 7: Merge histories
    print("=" * 50)
    print("Example 7: Merge Multiple Histories")
    print("=" * 50)

    from lexilux import merge_histories

    history1 = ChatHistory.from_messages(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    )

    history2 = ChatHistory.from_messages(
        [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]
    )

    merged = merge_histories(history1, history2)

    print(f"Merged history has {len(merged.messages)} messages:")
    for msg in merged.messages:
        print(f"  [{msg.role}]: {msg.content}")

    # Example 8: Statistics
    print("\n" + "=" * 50)
    print("Example 8: Conversation Statistics")
    print("=" * 50)

    from lexilux import get_statistics

    stats = get_statistics(history)

    print(f"Total messages: {stats['message_count']}")
    print(f"User messages: {stats['user_message_count']}")
    print(f"Assistant messages: {stats['assistant_message_count']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens per message: {stats['avg_tokens_per_message']:.1f}")


if __name__ == "__main__":
    main()
