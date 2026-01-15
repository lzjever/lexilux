"""Test ChatHistory deep copy protection."""

from lexilux.chat.history import ChatHistory


def test_chat_history_deep_copy_protection():
    """External modification should not affect ChatHistory internal state."""
    # Create original message
    original_msg = {"role": "user", "content": "hello"}

    # Create history with the message
    history = ChatHistory(messages=[original_msg])

    # External modification
    original_msg["content"] = "hacked"

    # ChatHistory internal state should not be affected
    assert history.messages[0]["content"] == "hello", (
        "External modification affected internal state"
    )
    assert len(history.messages) == 1


def test_chat_history_deep_copy_with_nested_messages():
    """Deep copy should protect nested message structures."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    history = ChatHistory(messages=messages)

    # Modify original list
    messages.append({"role": "user", "content": "This should not appear"})
    messages[0]["content"] = "Modified"

    # History should be unchanged
    assert len(history.messages) == 3
    assert history.messages[0]["content"] == "Hello"
    assert history.messages[2]["content"] == "How are you?"


def test_chat_history_from_messages_deep_copy():
    """from_messages should also deep copy."""
    original = [{"role": "user", "content": "test"}]
    history = ChatHistory.from_messages(original)

    # Modify original
    original[0]["content"] = "modified"

    # History should be unchanged
    assert history.messages[0]["content"] == "test"


def test_chat_history_limitation_documented():
    """
    Document the limitation: accessing via index returns a reference.

    NOTE: This is a Python limitation - lists store object references.
    Once you access history.messages[i], you get a reference to the dict,
    and modifying it will affect the internal state.

    Best practice: If you need to work with messages, use get_messages() or
    iterate rather than storing references.
    """
    history = ChatHistory(messages=[{"role": "user", "content": "original"}])

    # Get message and modify it
    msg = history.messages[0]
    msg["content"] = "modified"

    # This WILL affect the history (Python list behavior)
    assert history.messages[0]["content"] == "modified"

    # To avoid this, work with copies:
    history2 = ChatHistory(messages=[{"role": "user", "content": "original"}])
    msg_copy = history2.messages[0].copy()  # Explicit copy
    msg_copy["content"] = "modified"
    assert history2.messages[0]["content"] == "original"  # Unchanged
