#!/usr/bin/env python
"""
05 Chat vs ChatHistory - Understanding the Core Concepts

This example explains the difference between Chat and ChatHistory,
which is often confusing for new users.

Note: The old "Conversation" class has been deprecated and renamed to
_ResponseContinuer (internal API). Users should use chat.complete() instead.

Level: Core Concept
"""

from config_loader import get_chat_config, parse_args


def main():
    """Demonstrate the difference between Chat and ChatHistory."""
    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)  # noqa: F841
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {  # noqa: F841
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    # Import here to avoid linting issues since this is a demonstration file
    from lexilux import Chat, ChatHistory  # noqa: F401

    # Initialize Chat client (actual API calls commented out for demo)
    # chat = Chat(**config)  # noqa: F841

    # =========================================================================
    # Part 1: Chat - The HTTP Client (Stateless)
    # =========================================================================
    print("=" * 70)
    print("PART 1: Chat - The HTTP Client")
    print("=" * 70)
    print("""
Chat is an HTTP client. Each call is INDEPENDENT - it doesn't remember
previous conversations unless you explicitly pass the history.
""")

    print("Example 1: Two independent calls (Chat doesn't remember)\n")
    print("Call 1: chat('My name is Alice')")
    # result1 = chat("My name is Alice")
    # print(f"Response: {result1.text[:50]}...")

    print("\nCall 2: chat('What is my name?')")
    print("Response: [AI doesn't know your name - no history!]")
    print("""
Why? Because Chat is stateless. Each call is a fresh request.
""")

    # =========================================================================
    # Part 2: ChatHistory - Managing Conversation State
    # =========================================================================
    print("=" * 70)
    print("PART 2: ChatHistory - Managing Conversation State")
    print("=" * 70)
    print("""
ChatHistory is a data container that stores your conversation.
You must manually manage it and pass it to Chat.
""")

    print("\nExample 2: Multi-turn conversation with ChatHistory\n")

    # Create history with system message
    history = ChatHistory(system="You are a friendly assistant.")

    # First turn
    print("User: My name is Alice")
    history.add_user("My name is Alice")

    # result = chat(history.get_messages())
    # history.add_assistant(result.text)
    # print(f"AI: {result.text[:50]}...")

    # Simulate response
    history.add_assistant("Nice to meet you, Alice!")
    print("AI: Nice to meet you, Alice!")

    # Second turn - AI remembers!
    print("\nUser: What is my name?")
    history.add_user("What is my name?")

    # result = chat(history.get_messages())
    print("AI: [Now AI knows your name is Alice!]")
    print("""
Key insight: ChatHistory maintains state, Chat does not.
You pass history.get_messages() to Chat for context.
""")

    # =========================================================================
    # Part 3: Handling Long Responses (Auto-Continue)
    # =========================================================================
    print("=" * 70)
    print("PART 3: Handling Long Responses")
    print("=" * 70)
    print("""
When a response is truncated (finish_reason == "length"), use chat.complete()
to automatically continue generation until done.
""")

    print("\nExample 3: Using chat.complete() for long content\n")

    print("--- The Easy Way: chat.complete() ---")
    print("""
result = chat.complete(
    "Write a 1000-word essay about AI",
    max_tokens=100,      # Small limit to trigger truncation
    max_continues=10,    # Auto-continue up to 10 times
)
# chat.complete() handles everything automatically!
""")

    print("--- The Old Way (DEPRECATED): Conversation class ---")
    print("""
# The "Conversation" class has been deprecated.
# It was confusingly named - it's not about conversation history!
# It's actually a utility for handling truncated responses.
# Use chat.complete() instead.
""")

    # =========================================================================
    # Part 4: Quick Reference
    # =========================================================================
    print("=" * 70)
    print("QUICK REFERENCE")
    print("=" * 70)
    print("""
┌─────────────┬────────────────┬─────────────┬─────────────────────────┐
│    Class    │      Role      │   Stateful? │      When to Use        │
├─────────────┼────────────────┼─────────────┼─────────────────────────┤
│ Chat        │ HTTP Client    │ No          │ All API calls           │
│ ChatHistory │ Data Container │ Yes         │ Multi-turn conversations│
└─────────────┴────────────────┴─────────────┴─────────────────────────┘

Chat Methods:
  - chat()       : Single request (may be truncated)
  - stream()     : Streaming response (may be truncated)
  - complete()   : Auto-continue if truncated (RECOMMENDED for long content)

Common Patterns:

1. Simple Q&A (no history):
   result = chat("What is Python?")

2. Multi-turn conversation:
   history = ChatHistory()
   history.add_user("Hello")
   result = chat(history.get_messages())
   history.add_assistant(result.text)

3. Long content generation:
   result = chat.complete("Write an essay", max_tokens=100)

4. All together:
   history = ChatHistory()
   result = chat.complete("Write a story", history=history)
""")

    # =========================================================================
    # Part 5: Common Mistakes
    # =========================================================================
    print("=" * 70)
    print("COMMON MISTAKES")
    print("=" * 70)
    print("""
❌ MISTAKE 1: Expecting Chat to remember
   chat("My name is Alice")
   chat("What's my name?")  # AI doesn't know!

✅ CORRECT: Use ChatHistory
   history = ChatHistory()
   history.add_user("My name is Alice")
   chat(history.get_messages())

❌ MISTAKE 2: Using deprecated "Conversation" class
   # The "Conversation" name was confusing (sounds like history management)
   # It's now deprecated - use chat.complete() instead

✅ CORRECT: Use chat.complete()
   result = chat.complete("Write a long story", max_tokens=50)
""")


if __name__ == "__main__":
    main()
