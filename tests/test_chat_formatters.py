"""
Comprehensive tests for ChatHistoryFormatter class.

Tests are written based on the public interface, not implementation details.
"""

import tempfile
from pathlib import Path

import pytest

from lexilux.chat import ChatHistory, ChatHistoryFormatter


class TestChatHistoryFormatterMarkdown:
    """Test ChatHistoryFormatter.to_markdown"""

    def test_to_markdown_basic(self):
        """Test basic markdown formatting"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi there")
        md = ChatHistoryFormatter.to_markdown(history)
        assert isinstance(md, str)
        assert "User" in md or "user" in md.lower()
        assert "Assistant" in md or "assistant" in md.lower()

    def test_to_markdown_with_system(self):
        """Test markdown with system message"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        md = ChatHistoryFormatter.to_markdown(history)
        assert "System" in md or "system" in md.lower()

    def test_to_markdown_round_numbers(self):
        """Test markdown with round numbers"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        history.add_user("Q2")
        history.add_assistant("A2")
        md = ChatHistoryFormatter.to_markdown(history, show_round_numbers=True)
        assert "Round" in md or "round" in md.lower()

    def test_to_markdown_no_round_numbers(self):
        """Test markdown without round numbers"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        md = ChatHistoryFormatter.to_markdown(history, show_round_numbers=False)
        # Should still contain content
        assert len(md) > 0

    def test_to_markdown_highlight_system(self):
        """Test markdown with system highlighting"""
        history = ChatHistory(system="You are helpful")
        md = ChatHistoryFormatter.to_markdown(history, highlight_system=True)
        assert "System Message" in md or "system" in md.lower()

    def test_to_markdown_no_highlight_system(self):
        """Test markdown without system highlighting"""
        history = ChatHistory(system="You are helpful")
        md = ChatHistoryFormatter.to_markdown(history, highlight_system=False)
        assert isinstance(md, str)

    def test_to_markdown_empty_history(self):
        """Test markdown with empty history"""
        history = ChatHistory()
        md = ChatHistoryFormatter.to_markdown(history)
        assert isinstance(md, str)

    def test_to_markdown_special_characters(self):
        """Test markdown with special characters"""
        history = ChatHistory()
        history.add_user("Text with **bold** and __italic__")
        md = ChatHistoryFormatter.to_markdown(history)
        # Should escape or handle special characters
        assert isinstance(md, str)


class TestChatHistoryFormatterHTML:
    """Test ChatHistoryFormatter.to_html"""

    def test_to_html_basic(self):
        """Test basic HTML formatting"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi there")
        html = ChatHistoryFormatter.to_html(history)
        assert isinstance(html, str)
        assert "<html>" in html
        assert "</html>" in html

    def test_to_html_with_system(self):
        """Test HTML with system message"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        html = ChatHistoryFormatter.to_html(history)
        assert "System" in html or "system" in html.lower()

    def test_to_html_theme_default(self):
        """Test HTML with default theme"""
        history = ChatHistory()
        history.add_user("Hello")
        html = ChatHistoryFormatter.to_html(history, theme="default")
        assert "<html>" in html

    def test_to_html_theme_dark(self):
        """Test HTML with dark theme"""
        history = ChatHistory()
        history.add_user("Hello")
        html = ChatHistoryFormatter.to_html(history, theme="dark")
        assert "<html>" in html
        # Dark theme should have different styling
        assert isinstance(html, str)

    def test_to_html_theme_minimal(self):
        """Test HTML with minimal theme"""
        history = ChatHistory()
        history.add_user("Hello")
        html = ChatHistoryFormatter.to_html(history, theme="minimal")
        assert "<html>" in html

    def test_to_html_round_numbers(self):
        """Test HTML with round numbers"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        html = ChatHistoryFormatter.to_html(history, show_round_numbers=True)
        assert "Round" in html or "round" in html.lower()

    def test_to_html_no_round_numbers(self):
        """Test HTML without round numbers"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        html = ChatHistoryFormatter.to_html(history, show_round_numbers=False)
        assert "<html>" in html

    def test_to_html_empty_history(self):
        """Test HTML with empty history"""
        history = ChatHistory()
        html = ChatHistoryFormatter.to_html(history)
        assert "<html>" in html

    def test_to_html_escapes_content(self):
        """Test HTML escapes content properly"""
        history = ChatHistory()
        history.add_user("<script>alert('xss')</script>")
        html = ChatHistoryFormatter.to_html(history)
        # Should escape HTML special characters
        assert "<script>" not in html or "&lt;" in html


class TestChatHistoryFormatterText:
    """Test ChatHistoryFormatter.to_text"""

    def test_to_text_basic(self):
        """Test basic text formatting"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi there")
        text = ChatHistoryFormatter.to_text(history)
        assert isinstance(text, str)
        assert "User" in text or "user" in text.lower()

    def test_to_text_with_system(self):
        """Test text with system message"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        text = ChatHistoryFormatter.to_text(history)
        assert "SYSTEM" in text or "System" in text or "system" in text.lower()

    def test_to_text_round_numbers(self):
        """Test text with round numbers"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        text = ChatHistoryFormatter.to_text(history, show_round_numbers=True)
        assert "Round" in text or "round" in text.lower()

    def test_to_text_width(self):
        """Test text with custom width"""
        history = ChatHistory()
        history.add_user("This is a long message that should be wrapped")
        text = ChatHistoryFormatter.to_text(history, width=20)
        assert isinstance(text, str)

    def test_to_text_empty_history(self):
        """Test text with empty history"""
        history = ChatHistory()
        text = ChatHistoryFormatter.to_text(history)
        assert isinstance(text, str)


class TestChatHistoryFormatterJSON:
    """Test ChatHistoryFormatter.to_json"""

    def test_to_json_basic(self):
        """Test basic JSON formatting"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi")
        json_str = ChatHistoryFormatter.to_json(history)
        assert isinstance(json_str, str)
        assert "messages" in json_str

    def test_to_json_with_indent(self):
        """Test JSON with indentation"""
        history = ChatHistory()
        history.add_user("Hello")
        json_str = ChatHistoryFormatter.to_json(history, indent=2)
        assert isinstance(json_str, str)
        # Indented JSON should have newlines
        assert "\n" in json_str

    def test_to_json_empty_history(self):
        """Test JSON with empty history"""
        history = ChatHistory()
        json_str = ChatHistoryFormatter.to_json(history)
        assert isinstance(json_str, str)
        assert "messages" in json_str


class TestChatHistoryFormatterSave:
    """Test ChatHistoryFormatter.save"""

    def test_save_markdown_auto(self):
        """Test save with auto format detection (.md)"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = f.name
        try:
            ChatHistoryFormatter.save(history, temp_path)
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert len(content) > 0
        finally:
            Path(temp_path).unlink()

    def test_save_html_auto(self):
        """Test save with auto format detection (.html)"""
        history = ChatHistory()
        history.add_user("Hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            temp_path = f.name
        try:
            ChatHistoryFormatter.save(history, temp_path)
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert "<html>" in content
        finally:
            Path(temp_path).unlink()

    def test_save_text_auto(self):
        """Test save with auto format detection (.txt)"""
        history = ChatHistory()
        history.add_user("Hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name
        try:
            ChatHistoryFormatter.save(history, temp_path)
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert len(content) > 0
        finally:
            Path(temp_path).unlink()

    def test_save_json_auto(self):
        """Test save with auto format detection (.json)"""
        history = ChatHistory()
        history.add_user("Hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            ChatHistoryFormatter.save(history, temp_path)
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert "messages" in content
        finally:
            Path(temp_path).unlink()

    def test_save_explicit_format(self):
        """Test save with explicit format"""
        history = ChatHistory()
        history.add_user("Hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            temp_path = f.name
        try:
            ChatHistoryFormatter.save(history, temp_path, format="markdown")
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert len(content) > 0
        finally:
            Path(temp_path).unlink()

    def test_save_with_options(self):
        """Test save with formatting options"""
        history = ChatHistory()
        history.add_user("Hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            temp_path = f.name
        try:
            ChatHistoryFormatter.save(history, temp_path, theme="dark")
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert "<html>" in content
        finally:
            Path(temp_path).unlink()

    def test_save_unknown_format_raises(self):
        """Test save with unknown format raises error"""
        history = ChatHistory()
        history.add_user("Hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unknown format"):
                ChatHistoryFormatter.save(history, temp_path, format="unknown_format")
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()


class TestChatHistoryFormatterEdgeCases:
    """Test ChatHistoryFormatter edge cases"""

    def test_to_markdown_empty_content(self):
        """Test markdown with empty content"""
        history = ChatHistory()
        history.add_user("")
        history.add_assistant("")
        md = ChatHistoryFormatter.to_markdown(history)
        assert isinstance(md, str)

    def test_to_html_empty_content(self):
        """Test HTML with empty content"""
        history = ChatHistory()
        history.add_user("")
        html = ChatHistoryFormatter.to_html(history)
        assert "<html>" in html

    def test_to_text_empty_content(self):
        """Test text with empty content"""
        history = ChatHistory()
        history.add_user("")
        text = ChatHistoryFormatter.to_text(history)
        assert isinstance(text, str)

    def test_to_markdown_multiple_rounds(self):
        """Test markdown with many rounds"""
        history = ChatHistory()
        for i in range(10):
            history.add_user(f"Q{i}")
            history.add_assistant(f"A{i}")
        md = ChatHistoryFormatter.to_markdown(history)
        assert "Round" in md or "round" in md.lower()

    def test_to_html_multiple_rounds(self):
        """Test HTML with many rounds"""
        history = ChatHistory()
        for i in range(10):
            history.add_user(f"Q{i}")
            history.add_assistant(f"A{i}")
        html = ChatHistoryFormatter.to_html(history)
        assert "<html>" in html

    def test_to_text_multiple_rounds(self):
        """Test text with many rounds"""
        history = ChatHistory()
        for i in range(10):
            history.add_user(f"Q{i}")
            history.add_assistant(f"A{i}")
        text = ChatHistoryFormatter.to_text(history)
        assert isinstance(text, str)
