#!/usr/bin/env python
"""
Chat History Formatting Demo

Demonstrates how to format and export conversation history in multiple formats:
Markdown, HTML, plain text, and JSON.
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat
from lexilux.chat import ChatHistory, ChatHistoryFormatter


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
            "  2. Or specify a config file: python examples/chat_formatting_demo.py --config /path/to/config.json"
        )
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    return Chat(**config)


def demo_markdown_formatting():
    """Demonstrate Markdown formatting"""
    print("=" * 70)
    print("Demo 1: Markdown Formatting")
    print("=" * 70)

    chat = get_chat_client()

    # Build conversation
    result = chat("What is Python?", system="You are a helpful assistant")
    history = ChatHistory.from_chat_result("What is Python?", result)

    # Format as Markdown
    print("\n1. Basic Markdown:")
    md = ChatHistoryFormatter.to_markdown(history)
    print(md[:200] + "..." if len(md) > 200 else md)

    # Markdown without round numbers
    print("\n2. Markdown without round numbers:")
    md2 = ChatHistoryFormatter.to_markdown(history, show_round_numbers=False)
    print(md2[:200] + "..." if len(md2) > 200 else md2)


def demo_html_formatting():
    """Demonstrate HTML formatting"""
    print("\n" + "=" * 70)
    print("Demo 2: HTML Formatting")
    print("=" * 70)

    chat = get_chat_client()

    # Build conversation
    result = chat("Explain recursion", system="You are a helpful assistant")
    history = ChatHistory.from_chat_result("Explain recursion", result)

    # Format as HTML
    print("\n1. Basic HTML:")
    html = ChatHistoryFormatter.to_html(history)
    print(f"   HTML length: {len(html)} characters")
    print(f"   First 200 chars: {html[:200]}...")

    # HTML with dark theme
    print("\n2. HTML with dark theme:")
    html_dark = ChatHistoryFormatter.to_html(history, theme="dark")
    print(f"   HTML length: {len(html_dark)} characters")


def demo_text_formatting():
    """Demonstrate plain text formatting"""
    print("\n" + "=" * 70)
    print("Demo 3: Plain Text Formatting")
    print("=" * 70)

    chat = get_chat_client()

    # Build conversation
    result = chat("What is a list comprehension?", system="You are a helpful assistant")
    history = ChatHistory.from_chat_result("What is a list comprehension?", result)

    # Format as text
    print("\n1. Basic text:")
    text = ChatHistoryFormatter.to_text(history)
    print(text[:300] + "..." if len(text) > 300 else text)

    # Text with custom width
    print("\n2. Text with custom width (60):")
    text_narrow = ChatHistoryFormatter.to_text(history, width=60)
    print(text_narrow[:300] + "..." if len(text_narrow) > 300 else text_narrow)


def demo_file_saving():
    """Demonstrate saving to files"""
    print("\n" + "=" * 70)
    print("Demo 4: Saving to Files")
    print("=" * 70)

    chat = get_chat_client()

    # Build conversation
    result = chat("Explain decorators in Python", system="You are a helpful assistant")
    history = ChatHistory.from_chat_result("Explain decorators in Python", result)

    # Save in different formats (commented out to avoid creating files in demo)
    print("\n1. Save as Markdown:")
    print("   ChatHistoryFormatter.save(history, 'conversation.md')")
    # ChatHistoryFormatter.save(history, "conversation.md")

    print("\n2. Save as HTML with dark theme:")
    print("   ChatHistoryFormatter.save(history, 'conversation.html', theme='dark')")
    # ChatHistoryFormatter.save(history, "conversation.html", theme="dark")

    print("\n3. Save as plain text:")
    print("   ChatHistoryFormatter.save(history, 'conversation.txt')")
    # ChatHistoryFormatter.save(history, "conversation.txt")

    print("\n4. Save as JSON:")
    print("   ChatHistoryFormatter.save(history, 'conversation.json')")
    # ChatHistoryFormatter.save(history, "conversation.json")


def main():
    """Main function to run all demos"""
    print("\n" + "=" * 70)
    print("Chat History Formatting Demo")
    print("=" * 70)
    print("\nThis demo shows how to format and export conversation history.")
    print("Note: Replace 'your-api-key' with your actual API key to run.")
    print("\n" + "=" * 70)

    try:
        demo_markdown_formatting()
        demo_html_formatting()
        demo_text_formatting()
        demo_file_saving()

        print("\n" + "=" * 70)
        print("All demos completed!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: This demo requires a valid API key and endpoint.")
        print("Please update the API credentials in the code.")


if __name__ == "__main__":
    main()

