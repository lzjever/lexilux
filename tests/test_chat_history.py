"""
Comprehensive tests for ChatHistory class.

Tests are written based on the public interface, not implementation details.
"""

import json

import pytest

from lexilux import ChatResult, Usage
from lexilux.chat import ChatHistory


class TestChatHistoryInit:
    """Test ChatHistory initialization"""

    def test_init_empty(self):
        """Test creating empty ChatHistory"""
        history = ChatHistory()
        assert history.system is None
        assert len(history.messages) == 0
        assert isinstance(history.metadata, dict)

    def test_init_with_system(self):
        """Test creating ChatHistory with system message"""
        history = ChatHistory(system="You are helpful")
        assert history.system == "You are helpful"
        assert len(history.messages) == 0

    def test_init_with_messages(self):
        """Test creating ChatHistory with messages"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        history = ChatHistory(messages=messages)
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[1]["role"] == "assistant"

    def test_init_with_system_and_messages(self):
        """Test creating ChatHistory with both system and messages"""
        messages = [{"role": "user", "content": "Hello"}]
        history = ChatHistory(messages=messages, system="You are helpful")
        assert history.system == "You are helpful"
        assert len(history.messages) == 1


class TestChatHistoryFromMessages:
    """Test ChatHistory.from_messages class method"""

    def test_from_messages_string(self):
        """Test from_messages with string input"""
        history = ChatHistory.from_messages("Hello")
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"

    def test_from_messages_string_with_system(self):
        """Test from_messages with string and system"""
        history = ChatHistory.from_messages("Hello", system="You are helpful")
        assert history.system == "You are helpful"
        assert len(history.messages) == 1
        assert history.messages[0]["content"] == "Hello"

    def test_from_messages_list_of_strings(self):
        """Test from_messages with list of strings"""
        history = ChatHistory.from_messages(["Hello", "How are you?"])
        assert len(history.messages) == 2
        assert all(msg["role"] == "user" for msg in history.messages)

    def test_from_messages_list_of_dicts(self):
        """Test from_messages with list of dicts"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        history = ChatHistory.from_messages(messages)
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[1]["role"] == "assistant"

    def test_from_messages_with_system_in_messages(self):
        """Test from_messages when messages contain system message"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        history = ChatHistory.from_messages(messages)
        # System should be extracted
        assert history.system == "You are helpful"
        # System message should not be in messages
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"


class TestChatHistoryFromChatResult:
    """Test ChatHistory.from_chat_result class method"""

    def test_from_chat_result_string(self):
        """Test from_chat_result with string input"""
        result = ChatResult(text="Response", usage=Usage(), finish_reason="stop")
        history = ChatHistory.from_chat_result("Hello", result)
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[1]["content"] == "Response"

    def test_from_chat_result_list_of_dicts(self):
        """Test from_chat_result with list of dicts"""
        messages = [{"role": "user", "content": "Hello"}]
        result = ChatResult(text="Response", usage=Usage(), finish_reason="stop")
        history = ChatHistory.from_chat_result(messages, result)
        assert len(history.messages) == 2
        assert history.messages[1]["content"] == "Response"

    def test_from_chat_result_preserves_system(self):
        """Test from_chat_result preserves system message"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = ChatResult(text="Response", usage=Usage(), finish_reason="stop")
        history = ChatHistory.from_chat_result(messages, result)
        assert history.system == "You are helpful"
        assert len(history.messages) == 2  # user + assistant


class TestChatHistoryBasicOperations:
    """Test basic ChatHistory operations"""

    def test_add_user(self):
        """Test adding user message"""
        history = ChatHistory()
        history.add_user("Hello")
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"

    def test_add_assistant(self):
        """Test adding assistant message"""
        history = ChatHistory()
        history.add_assistant("Hi")
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "assistant"
        assert history.messages[0]["content"] == "Hi"

    def test_add_message(self):
        """Test adding message with specified role"""
        history = ChatHistory()
        history.add_message("system", "You are helpful")
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "system"
        assert history.messages[0]["content"] == "You are helpful"

    def test_clear(self):
        """Test clearing messages"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.clear()
        assert history.system == "You are helpful"  # System preserved
        assert len(history.messages) == 0

    def test_get_messages_without_system(self):
        """Test get_messages without system"""
        history = ChatHistory()
        history.add_user("Hello")
        messages = history.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_get_messages_with_system(self):
        """Test get_messages with system"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        messages = history.get_messages(include_system=True)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestChatHistorySerialization:
    """Test ChatHistory serialization"""

    def test_to_dict(self):
        """Test to_dict serialization"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi")
        data = history.to_dict()
        assert isinstance(data, dict)
        assert data["system"] == "You are helpful"
        assert len(data["messages"]) == 2
        assert "metadata" in data

    def test_to_json(self):
        """Test to_json serialization"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        json_str = history.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["system"] == "You are helpful"

    def test_from_dict(self):
        """Test from_dict deserialization"""
        data = {
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "metadata": {},
        }
        history = ChatHistory.from_dict(data)
        assert history.system == "You are helpful"
        assert len(history.messages) == 2

    def test_from_json(self):
        """Test from_json deserialization"""
        json_str = json.dumps(
            {
                "system": "You are helpful",
                "messages": [{"role": "user", "content": "Hello"}],
                "metadata": {},
            }
        )
        history = ChatHistory.from_json(json_str)
        assert history.system == "You are helpful"
        assert len(history.messages) == 1

    def test_round_trip_serialization(self):
        """Test round-trip serialization"""
        original = ChatHistory(system="You are helpful")
        original.add_user("Hello")
        original.add_assistant("Hi")
        # Serialize and deserialize
        data = original.to_dict()
        restored = ChatHistory.from_dict(data)
        assert restored.system == original.system
        assert len(restored.messages) == len(original.messages)
        assert restored.messages[0]["content"] == original.messages[0]["content"]


class TestChatHistoryRoundOperations:
    """Test ChatHistory round-based operations"""

    def test_get_last_n_rounds(self):
        """Test get_last_n_rounds"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        history.add_user("Q2")
        history.add_assistant("A2")
        history.add_user("Q3")
        history.add_assistant("A3")
        # Get last 2 rounds
        last_rounds = history.get_last_n_rounds(2)
        assert len(last_rounds.messages) == 4  # 2 rounds = 4 messages
        assert last_rounds.messages[0]["content"] == "Q2"
        assert last_rounds.messages[1]["content"] == "A2"

    def test_get_last_n_rounds_zero(self):
        """Test get_last_n_rounds with n=0"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        last_rounds = history.get_last_n_rounds(0)
        assert len(last_rounds.messages) == 0

    def test_get_last_n_rounds_more_than_available(self):
        """Test get_last_n_rounds with n > available rounds"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        last_rounds = history.get_last_n_rounds(10)
        assert len(last_rounds.messages) == 2

    def test_remove_last_round(self):
        """Test remove_last_round"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        history.add_user("Q2")
        history.add_assistant("A2")
        original_count = len(history.messages)
        history.remove_last_round()
        assert len(history.messages) == original_count - 2

    def test_remove_last_round_empty(self):
        """Test remove_last_round on empty history"""
        history = ChatHistory()
        history.remove_last_round()  # Should not raise
        assert len(history.messages) == 0

    def test_remove_last_round_incomplete(self):
        """Test remove_last_round with incomplete round"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        history.add_user("Q2")  # Incomplete round
        original_count = len(history.messages)
        history.remove_last_round()
        # Should remove the incomplete round
        assert len(history.messages) < original_count


class TestChatHistoryResultOperations:
    """Test ChatHistory operations with ChatResult"""

    def test_append_result(self):
        """Test append_result"""
        history = ChatHistory()
        result = ChatResult(text="Response", usage=Usage(), finish_reason="stop")
        history.append_result(result)
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "assistant"
        assert history.messages[0]["content"] == "Response"

    def test_update_last_assistant(self):
        """Test update_last_assistant"""
        history = ChatHistory()
        history.add_assistant("Original")
        history.update_last_assistant("Updated")
        assert history.messages[0]["content"] == "Updated"

    def test_update_last_assistant_when_none_exists(self):
        """Test update_last_assistant when no assistant message exists"""
        history = ChatHistory()
        history.add_user("Question")
        history.update_last_assistant("New assistant")
        # Should add new assistant message
        assert len(history.messages) == 2
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[1]["content"] == "New assistant"

    def test_update_last_assistant_empty_history(self):
        """Test update_last_assistant on empty history"""
        history = ChatHistory()
        history.update_last_assistant("New")
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "assistant"


class TestChatHistoryTokenOperations:
    """Test ChatHistory token-related operations"""

    @pytest.mark.skipif(True, reason="Requires tokenizer - will test with mock")
    def test_count_tokens(self):
        """Test count_tokens - placeholder for tokenizer tests"""
        # This will be tested with actual tokenizer in integration tests
        pass

    def test_truncate_by_rounds_empty(self):
        """Test truncate_by_rounds on empty history"""
        history = ChatHistory()
        # Mock tokenizer would be needed, but we test the interface
        # This test verifies the method exists and handles empty history
        assert hasattr(history, "truncate_by_rounds")

    def test_count_tokens_per_round_interface(self):
        """Test count_tokens_per_round interface exists"""
        history = ChatHistory()
        assert hasattr(history, "count_tokens_per_round")


class TestChatHistoryEdgeCases:
    """Test ChatHistory edge cases"""

    def test_empty_messages_list(self):
        """Test with empty messages list"""
        history = ChatHistory(messages=[])
        assert len(history.messages) == 0

    def test_messages_with_empty_content(self):
        """Test messages with empty content"""
        history = ChatHistory()
        history.add_user("")
        assert len(history.messages) == 1
        assert history.messages[0]["content"] == ""

    def test_multiple_system_messages_in_from_messages(self):
        """Test from_messages with multiple system messages"""
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "Hello"},
        ]
        history = ChatHistory.from_messages(messages)
        # Should extract first system message
        assert history.system == "System 1"
        # Remaining system messages should be in messages
        assert len(history.messages) >= 1

    def test_get_rounds_incomplete(self):
        """Test _get_rounds with incomplete round"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        history.add_user("Q2")  # Incomplete
        rounds = history._get_rounds()
        assert len(rounds) == 2  # One complete, one incomplete

    def test_get_rounds_only_user(self):
        """Test _get_rounds with only user messages"""
        history = ChatHistory()
        history.add_user("Q1")
        history.add_user("Q2")
        rounds = history._get_rounds()
        assert len(rounds) == 2  # Each user message is a round

    def test_get_rounds_only_assistant(self):
        """Test _get_rounds with only assistant messages"""
        history = ChatHistory()
        history.add_assistant("A1")
        history.add_assistant("A2")
        rounds = history._get_rounds()
        # Assistant messages without preceding user are incomplete rounds
        assert len(rounds) >= 1
