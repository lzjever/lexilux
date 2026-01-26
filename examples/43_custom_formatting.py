#!/usr/bin/env python
"""
43 Custom Formatting - Export Conversations

Learn how to format and export conversations in various formats:
Markdown, HTML, JSON, plain text, and custom formats.

Level: Expert Topic
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat, ChatHistory, ChatHistoryFormatter, Conversation


def main():
    """Demonstrate custom formatting features."""
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

    # Create a sample conversation
    print("Creating sample conversation...\n")
    conv = Conversation()
    conv.chat(chat, "What is Python?")
    conv.chat(chat, "Give me 3 key features")
    conv.chat(chat, "That's helpful, thanks!")

    history = conv.to_history()

    # Example 1: Markdown formatting
    print("=" * 50)
    print("Example 1: Markdown Format")
    print("=" * 50)

    formatter = ChatHistoryFormatter()
    markdown = formatter.format_markdown(history)

    print(markdown)
    print()

    # Example 2: HTML formatting
    print("=" * 50)
    print("Example 2: HTML Format")
    print("=" * 50)

    html = formatter.format_html(history, theme="light")

    print(html[:300] + "...\n")

    # Example 3: Plain text formatting
    print("=" * 50)
    print("Example 3: Plain Text Format")
    print("=" * 50)

    text = formatter.format_text(history, width=60)

    print(text)
    print()

    # Example 4: JSON format
    print("=" * 50)
    print("Example 4: JSON Format")
    print("=" * 50)

    json_str = formatter.format_json(history)

    print(json_str[:200] + "...\n")

    # Example 5: Custom formatting
    print("=" * 50)
    print("Example 5: Custom Formatter")
    print("=" * 50)

    class CustomFormatter:
        """Custom conversation formatter."""

        def format(self, history: ChatHistory) -> str:
            """Format as a chat log."""
            lines = ["=" * 60]
            lines.append("CONVERSATION LOG")
            lines.append("=" * 60)
            lines.append("")

            for i, msg in enumerate(history.messages, 1):
                role = msg.role.upper()
                content = msg.content

                lines.append(f"[{i}] {role}:")
                lines.append("-" * 40)
                lines.append(content)
                lines.append("")

            # Add summary
            lines.append("=" * 60)
            lines.append("SUMMARY")
            lines.append("=" * 60)
            lines.append(f"Total messages: {len(history.messages)}")
            lines.append(f"Total tokens: {history.usage.total_tokens}")

            return "\n".join(lines)

    custom_formatter = CustomFormatter()
    custom_output = custom_formatter.format(history)

    print(custom_output)
    print()

    # Example 6: Save to file
    print("=" * 50)
    print("Example 6: Save to File")
    print("=" * 50)

    import tempfile
    from pathlib import Path

    # Create a temp directory for demo
    temp_dir = Path(tempfile.mkdtemp())

    # Save in different formats
    files = {
        "conversation.md": formatter.format_markdown(history),
        "conversation.html": formatter.format_html(history),
        "conversation.txt": formatter.format_text(history),
        "conversation.json": formatter.format_json(history),
    }

    for filename, content in files.items():
        filepath = temp_dir / filename
        filepath.write_text(content)
        print(f"✓ Saved: {filepath}")

    print(f"\nAll files saved to: {temp_dir}\n")

    # Example 7: Format with theme
    print("=" * 50)
    print("Example 7: HTML with Different Themes")
    print("=" * 50)

    themes = ["light", "dark", "auto"]

    for theme in themes:
        html = formatter.format_html(history, theme=theme)
        # Count characters as a simple check
        print(f"Theme '{theme}': {len(html)} characters")

    print()

    # Example 8: Format with/without round numbers
    print("=" * 50)
    print("Example 8: Markdown with/without Round Numbers")
    print("=" * 50)

    print("With round numbers:")
    with_rounds = formatter.format_markdown(history, include_rounds=True)
    print(with_rounds[:200] + "...\n")

    print("Without round numbers:")
    without_rounds = formatter.format_markdown(history, include_rounds=False)
    print(without_rounds[:200] + "...\n")

    # Example 9: Filtering before formatting
    print("=" * 50)
    print("Example 9: Filter Messages Before Formatting")
    print("=" * 50)

    from lexilux import filter_by_role

    # Format only user messages
    user_msgs = filter_by_role(history, "user")
    user_history = ChatHistory(messages=user_msgs)

    print("User messages only:")
    print(formatter.format_markdown(user_history))

    print("\n✓ Custom formatting completed!")


if __name__ == "__main__":
    main()
