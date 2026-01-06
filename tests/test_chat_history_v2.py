"""
Comprehensive tests for ChatHistory v2.0 (MutableSequence protocol).

Tests are written based on the public interface specification, not implementation details.
Tests challenge the business logic and verify correct behavior according to the API contract.

v2.0 API changes:
1. ChatHistory implements MutableSequence protocol
2. Supports indexing, slicing, iteration, length, membership
3. Added query and modification methods
4. Added clone() and __add__() methods
"""

import pytest

from lexilux import ChatHistory, ChatResult, Usage


class TestChatHistoryMutableSequence:
    """Test ChatHistory as MutableSequence"""

    def test_len(self):
        """Test __len__() method"""
        history = ChatHistory()
        assert len(history) == 0

        history.add_user("Hello")
        assert len(history) == 1

        history.add_assistant("Hi!")
        assert len(history) == 2

    def test_getitem_index(self):
        """Test __getitem__() with index"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        msg = history[0]
        assert isinstance(msg, dict)
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

        msg = history[1]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hi!"

    def test_getitem_slice(self):
        """Test __getitem__() with slice (returns new ChatHistory)"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")
        history.add_assistant("I'm fine")

        # Get first 2 messages
        first_2 = history[:2]
        assert isinstance(first_2, ChatHistory)
        assert len(first_2) == 2
        assert first_2[0]["content"] == "Hello"
        assert first_2[1]["content"] == "Hi!"

        # Original should be unchanged
        assert len(history) == 4

        # Get last 2 messages
        last_2 = history[-2:]
        assert len(last_2) == 2
        assert last_2[0]["content"] == "How are you?"
        assert last_2[1]["content"] == "I'm fine"

        # Get middle messages
        middle = history[1:3]
        assert len(middle) == 2
        assert middle[0]["content"] == "Hi!"
        assert middle[1]["content"] == "How are you?"

    def test_setitem_index(self):
        """Test __setitem__() with index"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        # Replace first message
        history[0] = {"role": "user", "content": "Updated"}
        assert history[0]["content"] == "Updated"

        # Replace second message
        history[1] = {"role": "assistant", "content": "Updated response"}
        assert history[1]["content"] == "Updated response"

    def test_setitem_slice(self):
        """Test __setitem__() with slice"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        # Replace first 2 messages
        history[:2] = [
            {"role": "user", "content": "New 1"},
            {"role": "assistant", "content": "New 2"},
        ]
        assert history[0]["content"] == "New 1"
        assert history[1]["content"] == "New 2"
        assert history[2]["content"] == "How are you?"

    def test_delitem_index(self):
        """Test __delitem__() with index"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        # Delete first message
        del history[0]
        assert len(history) == 2
        assert history[0]["content"] == "Hi!"

    def test_delitem_slice(self):
        """Test __delitem__() with slice"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")
        history.add_assistant("I'm fine")

        # Delete first 2 messages
        del history[:2]
        assert len(history) == 2
        assert history[0]["content"] == "How are you?"
        assert history[1]["content"] == "I'm fine"

    def test_insert(self):
        """Test insert() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        # Insert at beginning
        history.insert(0, {"role": "system", "content": "You are helpful"})
        assert len(history) == 3
        assert history[0]["role"] == "system"
        assert history[1]["content"] == "Hello"

        # Insert in middle
        history.insert(2, {"role": "user", "content": "How are you?"})
        assert len(history) == 4
        assert history[2]["content"] == "How are you?"
        assert history[3]["content"] == "Hi!"

    def test_iter(self):
        """Test __iter__() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        messages = list(history)
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"
        assert messages[2]["content"] == "How are you?"

    def test_contains(self):
        """Test __contains__() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        # Check membership (by value comparison, as per list behavior)
        assert history.messages[0] in history
        assert history.messages[1] in history

        # New dict with same content should be in history (value comparison)
        # This is standard Python list behavior
        new_msg = {"role": "user", "content": "Hello"}
        assert new_msg in history  # Value comparison, not identity

        # Different content should not be in history
        different_msg = {"role": "user", "content": "Different"}
        assert different_msg not in history


class TestChatHistoryQueryMethods:
    """Test ChatHistory query methods"""

    def test_get_user_messages(self):
        """Test get_user_messages() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")
        history.add_assistant("I'm fine")

        user_msgs = history.get_user_messages()
        assert user_msgs == ["Hello", "How are you?"]

    def test_get_assistant_messages(self):
        """Test get_assistant_messages() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")
        history.add_assistant("I'm fine")

        assistant_msgs = history.get_assistant_messages()
        assert assistant_msgs == ["Hi!", "I'm fine"]

    def test_get_last_message(self):
        """Test get_last_message() method"""
        history = ChatHistory()
        assert history.get_last_message() is None

        history.add_user("Hello")
        last = history.get_last_message()
        assert last["content"] == "Hello"

        history.add_assistant("Hi!")
        last = history.get_last_message()
        assert last["content"] == "Hi!"

    def test_get_last_user_message(self):
        """Test get_last_user_message() method"""
        history = ChatHistory()
        assert history.get_last_user_message() is None

        history.add_user("Hello")
        assert history.get_last_user_message() == "Hello"

        history.add_assistant("Hi!")
        assert history.get_last_user_message() == "Hello"

        history.add_user("How are you?")
        assert history.get_last_user_message() == "How are you?"


class TestChatHistoryModificationMethods:
    """Test ChatHistory modification methods"""

    def test_remove_last(self):
        """Test remove_last() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        removed = history.remove_last()
        assert removed["content"] == "Hi!"
        assert len(history) == 1
        assert history[0]["content"] == "Hello"

        # Remove from empty
        history.clear()
        assert history.remove_last() is None

    def test_remove_at(self):
        """Test remove_at() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        # Remove middle message
        removed = history.remove_at(1)
        assert removed["content"] == "Hi!"
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "How are you?"

        # Remove out of range
        assert history.remove_at(10) is None
        assert history.remove_at(-10) is None

    def test_replace_at(self):
        """Test replace_at() method"""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        # Replace first message
        history.replace_at(0, "user", "Updated")
        assert history[0]["content"] == "Updated"

        # Replace second message
        history.replace_at(1, "assistant", "Updated response")
        assert history[1]["content"] == "Updated response"

        # Replace out of range should raise
        with pytest.raises(IndexError):
            history.replace_at(10, "user", "Invalid")

    def test_add_system(self):
        """Test add_system() method"""
        history = ChatHistory()
        assert history.system is None

        history.add_system("You are helpful")
        assert history.system == "You are helpful"

        # Update system message
        history.add_system("You are very helpful")
        assert history.system == "You are very helpful"


class TestChatHistoryClone:
    """Test ChatHistory clone() method"""

    def test_clone_creates_deep_copy(self):
        """Test that clone() creates a deep copy"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi!")

        cloned = history.clone()

        # Should be equal
        assert cloned.system == history.system
        assert len(cloned.messages) == len(history.messages)
        assert cloned.messages[0]["content"] == history.messages[0]["content"]

        # But different objects
        assert cloned is not history
        assert cloned.messages is not history.messages
        assert cloned.messages[0] is not history.messages[0]

        # Modifying clone should not affect original
        cloned.add_user("New message")
        assert len(cloned.messages) == 3
        assert len(history.messages) == 2

        cloned.messages[0]["content"] = "Modified"
        assert history.messages[0]["content"] == "Hello"  # Original unchanged


class TestChatHistoryAdd:
    """Test ChatHistory __add__() method"""

    def test_add_merges_histories(self):
        """Test that __add__() merges two histories"""
        history1 = ChatHistory(system="You are helpful")
        history1.add_user("Hello")
        history1.add_assistant("Hi!")

        history2 = ChatHistory()
        history2.add_user("How are you?")
        history2.add_assistant("I'm fine")

        combined = history1 + history2

        # Should be new ChatHistory
        assert isinstance(combined, ChatHistory)
        assert combined is not history1
        assert combined is not history2

        # Should have system from first history
        assert combined.system == "You are helpful"

        # Should have all messages
        assert len(combined.messages) == 4
        assert combined.messages[0]["content"] == "Hello"
        assert combined.messages[1]["content"] == "Hi!"
        assert combined.messages[2]["content"] == "How are you?"
        assert combined.messages[3]["content"] == "I'm fine"

        # Original histories should be unchanged
        assert len(history1.messages) == 2
        assert len(history2.messages) == 2


class TestChatHistoryExistingMethods:
    """Test existing ChatHistory methods still work"""

    def test_add_user(self):
        """Test add_user() method"""
        history = ChatHistory()
        history.add_user("Hello")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_add_assistant(self):
        """Test add_assistant() method"""
        history = ChatHistory()
        history.add_assistant("Hi!")
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert history[0]["content"] == "Hi!"

    def test_add_message(self):
        """Test add_message() method"""
        history = ChatHistory()
        history.add_message("system", "You are helpful")
        assert len(history) == 1
        assert history[0]["role"] == "system"
        assert history[0]["content"] == "You are helpful"

    def test_clear(self):
        """Test clear() method"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi!")

        history.clear()
        assert len(history) == 0
        assert history.system == "You are helpful"  # System preserved

    def test_get_messages(self):
        """Test get_messages() method"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi!")

        # With system
        messages = history.get_messages(include_system=True)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

        # Without system
        messages = history.get_messages(include_system=False)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_append_result(self):
        """Test append_result() method"""
        history = ChatHistory()
        history.add_user("Hello")

        result = ChatResult(
            text="Hi!",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

        history.append_result(result)
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi!"

