"""
Chat history formatters.

Provides formatting and export functionality for ChatHistory in multiple formats:
Markdown, HTML, plain text, and JSON.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from lexilux.chat.history import ChatHistory


class ChatHistoryFormatter:
    """
    Chat history formatter.

    Provides static methods to format ChatHistory into various output formats.
    """

    @staticmethod
    def to_markdown(
        history: ChatHistory,
        *,
        show_round_numbers: bool = True,
        show_timestamps: bool = False,
        highlight_system: bool = True,
    ) -> str:
        """
        Format history as Markdown.

        Args:
            history: ChatHistory instance to format.
            show_round_numbers: Whether to show round numbers. Default: True
            highlight_system: Whether to highlight system message. Default: True
            show_timestamps: Whether to show timestamps (if available). Default: False

        Returns:
            Markdown formatted string.

        Examples:
            >>> history = ChatHistory.from_chat_result("Hello", result)
            >>> md = ChatHistoryFormatter.to_markdown(history)
            >>> print(md)
        """
        lines = []
        messages = history.get_messages(include_system=True)

        round_num = 0
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # System message
            if role == "system":
                if highlight_system:
                    lines.append("## System Message")
                    lines.append("")
                    lines.append(f"*{content}*")
                else:
                    lines.append(f"**System:** {content}")
                lines.append("")
                continue

            # User message - start new round
            if role == "user":
                round_num += 1
                if show_round_numbers:
                    lines.append(f"### Round {round_num}")
                    lines.append("")
                lines.append("**User:**")
                lines.append("")
                # Escape markdown special characters in content
                content_escaped = content.replace("**", "\\*\\*").replace("__", "\\_\\_")
                lines.append(content_escaped)
                lines.append("")

            # Assistant message
            elif role == "assistant":
                lines.append("**Assistant:**")
                lines.append("")
                content_escaped = content.replace("**", "\\*\\*").replace("__", "\\_\\_")
                lines.append(content_escaped)
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_html(
        history: ChatHistory,
        *,
        theme: str = "default",
        show_round_numbers: bool = True,
        show_timestamps: bool = False,
    ) -> str:
        """
        Format history as HTML (beautiful and clear).

        Args:
            history: ChatHistory instance to format.
            theme: Theme name ("default", "dark", "minimal"). Default: "default"
            show_round_numbers: Whether to show round numbers. Default: True
            show_timestamps: Whether to show timestamps (if available). Default: False

        Returns:
            HTML formatted string with embedded CSS.

        Examples:
            >>> history = ChatHistory.from_chat_result("Hello", result)
            >>> html = ChatHistoryFormatter.to_html(history, theme="dark")
        """
        messages = history.get_messages(include_system=True)

        # CSS styles based on theme
        css_styles = {
            "default": """
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                       line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 900px; margin: 0 auto; background: white;
                            padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .system { background: #e3f2fd; padding: 15px; border-radius: 6px;
                         margin-bottom: 20px; border-left: 4px solid #2196f3; }
                .round { margin-bottom: 30px; border: 1px solid #e0e0e0;
                        border-radius: 6px; overflow: hidden; }
                .round-header { background: #fafafa; padding: 10px 15px;
                               font-weight: 600; color: #666; border-bottom: 1px solid #e0e0e0; }
                .message { padding: 15px; }
                .user { background: #f5f5f5; border-left: 4px solid #4caf50; }
                .assistant { background: #fff; border-left: 4px solid #2196f3; }
                .role { font-weight: 600; margin-bottom: 8px; color: #333; }
                .content { color: #444; white-space: pre-wrap; }
            """,
            "dark": """
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                       line-height: 1.6; margin: 0; padding: 20px; background: #1e1e1e; color: #e0e0e0; }
                .container { max-width: 900px; margin: 0 auto; background: #2d2d2d;
                            padding: 30px; border-radius: 8px; }
                .system { background: #1a237e; padding: 15px; border-radius: 6px;
                         margin-bottom: 20px; border-left: 4px solid #3f51b5; }
                .round { margin-bottom: 30px; border: 1px solid #404040;
                        border-radius: 6px; overflow: hidden; }
                .round-header { background: #333; padding: 10px 15px;
                               font-weight: 600; color: #aaa; border-bottom: 1px solid #404040; }
                .message { padding: 15px; }
                .user { background: #2d2d2d; border-left: 4px solid #4caf50; }
                .assistant { background: #252525; border-left: 4px solid #2196f3; }
                .role { font-weight: 600; margin-bottom: 8px; color: #fff; }
                .content { color: #e0e0e0; white-space: pre-wrap; }
            """,
            "minimal": """
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                       line-height: 1.8; margin: 0; padding: 20px; background: white; }
                .container { max-width: 900px; margin: 0 auto; padding: 20px; }
                .system { padding: 10px 0; margin-bottom: 20px; border-bottom: 1px solid #eee;
                         font-style: italic; color: #666; }
                .round { margin-bottom: 25px; }
                .round-header { font-weight: 600; color: #999; margin-bottom: 10px; font-size: 0.9em; }
                .message { margin-bottom: 15px; }
                .user { padding-left: 15px; border-left: 2px solid #4caf50; }
                .assistant { padding-left: 15px; border-left: 2px solid #2196f3; }
                .role { font-weight: 600; margin-bottom: 5px; color: #333; }
                .content { color: #444; white-space: pre-wrap; }
            """,
        }

        css = css_styles.get(theme, css_styles["default"])

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Chat History</title>",
            f"<style>{css}</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
        ]

        round_num = 0
        in_round = False

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # System message
            if role == "system":
                html_parts.append("<div class='system'>")
                html_parts.append(f"<strong>System:</strong> {html.escape(content)}")
                html_parts.append("</div>")
                continue

            # User message - start new round
            if role == "user":
                if in_round:
                    html_parts.append("</div>")  # Close previous round
                round_num += 1
                in_round = True
                html_parts.append("<div class='round'>")
                if show_round_numbers:
                    html_parts.append(f"<div class='round-header'>Round {round_num}</div>")
                html_parts.append("<div class='message user'>")
                html_parts.append("<div class='role'>User</div>")
                html_parts.append(f"<div class='content'>{html.escape(content)}</div>")
                html_parts.append("</div>")

            # Assistant message
            elif role == "assistant":
                html_parts.append("<div class='message assistant'>")
                html_parts.append("<div class='role'>Assistant</div>")
                html_parts.append(f"<div class='content'>{html.escape(content)}</div>")
                html_parts.append("</div>")
                html_parts.append("</div>")  # Close round
                in_round = False

        if in_round:
            html_parts.append("</div>")  # Close last round if incomplete

        html_parts.extend(["</div>", "</body>", "</html>"])
        return "\n".join(html_parts)

    @staticmethod
    def to_text(
        history: ChatHistory,
        *,
        show_round_numbers: bool = True,
        width: int = 80,
    ) -> str:
        """
        Format history as plain text (console-friendly).

        Args:
            history: ChatHistory instance to format.
            show_round_numbers: Whether to show round numbers. Default: True
            width: Text width for wrapping. Default: 80

        Returns:
            Plain text formatted string.

        Examples:
            >>> history = ChatHistory.from_chat_result("Hello", result)
            >>> text = ChatHistoryFormatter.to_text(history, width=100)
        """
        lines = []
        messages = history.get_messages(include_system=True)

        round_num = 0
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # System message
            if role == "system":
                lines.append("=" * width)
                lines.append("SYSTEM MESSAGE")
                lines.append("=" * width)
                lines.append(content)
                lines.append("")
                continue

            # User message - start new round
            if role == "user":
                round_num += 1
                if show_round_numbers:
                    lines.append("")
                    lines.append("-" * width)
                    lines.append(f"Round {round_num}")
                    lines.append("-" * width)
                else:
                    lines.append("-" * width)
                lines.append("User:")
                lines.append("")
                # Simple text wrapping (basic implementation)
                words = content.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= width:
                        current_line += (word + " ") if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                lines.append("")

            # Assistant message
            elif role == "assistant":
                lines.append("Assistant:")
                lines.append("")
                words = content.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= width:
                        current_line += (word + " ") if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_json(history: ChatHistory, **kwargs) -> str:
        """
        Format history as JSON (program-friendly).

        Args:
            history: ChatHistory instance to format.
            **kwargs: Additional arguments for json.dumps (e.g., indent=2).

        Returns:
            JSON formatted string.

        Examples:
            >>> history = ChatHistory.from_chat_result("Hello", result)
            >>> json_str = ChatHistoryFormatter.to_json(history, indent=2)
        """
        return history.to_json(**kwargs)

    @staticmethod
    def save(
        history: ChatHistory,
        filepath: str,
        format: str = "auto",
        **options: Any,
    ) -> None:
        """
        Save history to file (automatically selects format based on extension).

        Args:
            history: ChatHistory instance to save.
            filepath: Path to save file.
            format: Format to use ("auto", "markdown", "html", "text", "json").
                   If "auto", format is determined by file extension.
            **options: Additional options for formatters.

        Examples:
            >>> history = ChatHistory.from_chat_result("Hello", result)
            >>> ChatHistoryFormatter.save(history, "conversation.md")
            >>> ChatHistoryFormatter.save(history, "conversation.html", theme="dark")
            >>> ChatHistoryFormatter.save(history, "conversation.txt", width=100)
        """
        path = Path(filepath)

        # Auto-detect format from extension
        if format == "auto":
            ext = path.suffix.lower()
            if ext == ".md" or ext == ".markdown":
                format = "markdown"
            elif ext == ".html" or ext == ".htm":
                format = "html"
            elif ext == ".txt" or ext == ".text":
                format = "text"
            elif ext == ".json":
                format = "json"
            else:
                # Default to markdown if unknown
                format = "markdown"

        # Format content
        if format == "markdown":
            content = ChatHistoryFormatter.to_markdown(history, **options)
        elif format == "html":
            content = ChatHistoryFormatter.to_html(history, **options)
        elif format == "text":
            content = ChatHistoryFormatter.to_text(history, **options)
        elif format == "json":
            content = ChatHistoryFormatter.to_json(history, **options)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Write to file
        path.write_text(content, encoding="utf-8")
