"""
Chat history management.

Provides ChatHistory class for managing conversation history with automatic extraction,
serialization, token counting, and truncation capabilities.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lexilux.chat.models import ChatResult, MessagesLike
from lexilux.chat.utils import normalize_messages

if TYPE_CHECKING:
    from lexilux.tokenizer import Tokenizer


class ChatHistory:
    """
    Conversation history manager (automatic extraction, no manual maintenance required).

    ChatHistory can be automatically built from messages or Chat results, eliminating
    the need for manual history maintenance.

    Examples:
        # Auto-extract from Chat call
        >>> result = chat("Hello")
        >>> history = ChatHistory.from_chat_result("Hello", result)

        # Auto-extract from messages
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> history = ChatHistory.from_messages(messages)

        # Manual construction (optional)
        >>> history = ChatHistory(system="You are helpful")
        >>> history.add_user("What is Python?")
        >>> result = chat(history.get_messages())
        >>> history.append_result(result)
    """

    def __init__(
        self,
        messages: list[dict[str, str]] | None = None,
        system: str | None = None,
    ):
        """
        Initialize conversation history.

        Args:
            messages: Message list (optional, can be extracted from anywhere).
            system: System message (optional).
        """
        self.system = system
        self.messages: list[dict[str, str]] = messages or []
        self.metadata: dict[str, Any] = {}  # Metadata (timestamps, model, etc.)

    @classmethod
    def from_messages(cls, messages: MessagesLike, system: str | None = None) -> ChatHistory:
        """
        Automatically build from message list (supports all Chat-supported formats).

        Args:
            messages: Messages in various formats (str, list of str, list of dict).
            system: Optional system message.

        Returns:
            ChatHistory instance.

        Examples:
            >>> history = ChatHistory.from_messages("Hello")
            >>> history = ChatHistory.from_messages([{"role": "user", "content": "Hello"}])
        """
        normalized = normalize_messages(messages, system=system)
        # Extract system message(s) if present
        # Only extract the first system message, keep others in messages
        sys_msg = None
        if normalized and normalized[0].get("role") == "system":
            sys_msg = normalized[0]["content"]
            normalized = normalized[1:]
        return cls(messages=normalized, system=sys_msg)

    @classmethod
    def from_chat_result(cls, messages: MessagesLike, result: ChatResult) -> ChatHistory:
        """
        Automatically build complete history from Chat call and result.

        Args:
            messages: Messages sent to Chat (supports all formats).
            result: ChatResult from the API call.

        Returns:
            ChatHistory instance with complete conversation.

        Examples:
            >>> result = chat("Hello")
            >>> history = ChatHistory.from_chat_result("Hello", result)
        """
        normalized = normalize_messages(messages)
        # Extract system message if present
        sys_msg = None
        if normalized and normalized[0].get("role") == "system":
            sys_msg = normalized[0]["content"]
            normalized = normalized[1:]

        # Add assistant response
        history_messages = normalized.copy()
        history_messages.append({"role": "assistant", "content": result.text})

        return cls(messages=history_messages, system=sys_msg)

    @classmethod
    def from_dict(cls, data: dict) -> ChatHistory:
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary containing history data.

        Returns:
            ChatHistory instance.
        """
        return cls(
            messages=data.get("messages", []),
            system=data.get("system"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ChatHistory:
        """
        Deserialize from JSON string.

        Args:
            json_str: JSON string containing history data.

        Returns:
            ChatHistory instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def add_user(self, content: str) -> None:
        """Add user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Add assistant message."""
        self.messages.append({"role": "assistant", "content": content})

    def add_message(self, role: str, content: str) -> None:
        """Add message with specified role."""
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        """Clear all messages (keep system message)."""
        self.messages = []

    def get_messages(self, include_system: bool = True) -> list[dict[str, str]]:
        """
        Get messages list.

        Args:
            include_system: Whether to include system message.

        Returns:
            List of message dictionaries.
        """
        result = []
        if include_system and self.system:
            result.append({"role": "system", "content": self.system})
        result.extend(self.messages)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary containing history data.
        """
        return {
            "system": self.system,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    def to_json(self, **kwargs) -> str:
        """
        Serialize to JSON string.

        Args:
            **kwargs: Additional arguments for json.dumps.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), **kwargs)

    def count_tokens(self, tokenizer: Tokenizer) -> int:
        """
        Count total tokens in history.

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            Total token count.
        """
        messages = self.get_messages(include_system=True)
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            result = tokenizer(content)
            total += result.total_tokens or 0
        return total

    def count_tokens_per_round(self, tokenizer: Tokenizer) -> list[tuple[int, int]]:
        """
        Count tokens per round.

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            List of (round_index, tokens) tuples.
        """
        rounds = self._get_rounds()
        result = []
        for idx, round_messages in enumerate(rounds):
            round_tokens = 0
            for msg in round_messages:
                content = msg.get("content", "")
                token_result = tokenizer(content)
                round_tokens += token_result.total_tokens or 0
            result.append((idx, round_tokens))
        return result

    def truncate_by_rounds(
        self,
        tokenizer: Tokenizer,
        max_tokens: int,
        keep_system: bool = True,
    ) -> ChatHistory:
        """
        Truncate by rounds, keeping the most recent rounds within max_tokens limit.

        Args:
            tokenizer: Tokenizer instance.
            max_tokens: Maximum token count.
            keep_system: Whether to keep system message.

        Returns:
            New ChatHistory instance (does not modify original).
        """
        rounds = self._get_rounds()
        if not rounds:
            return ChatHistory(messages=[], system=self.system if keep_system else None)

        # Count tokens per round
        round_tokens = self.count_tokens_per_round(tokenizer)
        system_tokens = 0
        if keep_system and self.system:
            sys_result = tokenizer(self.system)
            system_tokens = sys_result.total_tokens or 0

        # Keep rounds from the end until we exceed max_tokens
        kept_rounds = []
        current_tokens = system_tokens
        for idx in range(len(rounds) - 1, -1, -1):
            round_token_count = round_tokens[idx][1]
            if current_tokens + round_token_count <= max_tokens:
                kept_rounds.insert(0, rounds[idx])
                current_tokens += round_token_count
            else:
                break

        # Rebuild messages
        new_messages = []
        for round_msgs in kept_rounds:
            new_messages.extend(round_msgs)

        return ChatHistory(
            messages=new_messages,
            system=self.system if keep_system else None,
        )

    def get_last_n_rounds(self, n: int) -> ChatHistory:
        """
        Get last N rounds.

        Args:
            n: Number of rounds to get.

        Returns:
            New ChatHistory instance with last N rounds.
        """
        rounds = self._get_rounds()
        if not rounds:
            return ChatHistory(messages=[], system=self.system)

        last_rounds = rounds[-n:] if n > 0 else []
        new_messages = []
        for round_msgs in last_rounds:
            new_messages.extend(round_msgs)

        return ChatHistory(messages=new_messages, system=self.system)

    def remove_last_round(self) -> None:
        """Remove the last round (user + assistant pair)."""
        rounds = self._get_rounds()
        if not rounds:
            return

        last_round = rounds[-1]
        for msg in last_round:
            if msg in self.messages:
                self.messages.remove(msg)

    def append_result(self, result: ChatResult) -> None:
        """Append ChatResult as assistant message."""
        self.add_assistant(result.text)

    def update_last_assistant(self, content: str) -> None:
        """Update the last assistant message content (useful for continue scenarios)."""
        # Find last assistant message
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                self.messages[i]["content"] = content
                return
        # If no assistant message found, add one
        self.add_assistant(content)

    def _get_rounds(self) -> list[list[dict[str, str]]]:
        """
        Get conversation rounds (user + assistant pairs).

        Returns:
            List of rounds, each round is a list of messages.
        """
        rounds = []
        current_round = []
        for msg in self.messages:
            role = msg.get("role")
            if role == "user":
                # Start new round
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            elif role == "assistant":
                # Add to current round
                current_round.append(msg)
                # Round complete
                rounds.append(current_round)
                current_round = []
        # Add incomplete round if exists
        if current_round:
            rounds.append(current_round)
        return rounds
