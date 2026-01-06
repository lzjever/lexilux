"""
Chat history management.

Provides ChatHistory class for managing conversation history with automatic extraction,
serialization, token counting, and truncation capabilities.
"""

from __future__ import annotations

import json
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lexilux.chat.models import ChatResult, MessagesLike
from lexilux.chat.utils import normalize_messages

if TYPE_CHECKING:
    from lexilux.tokenizer import Tokenizer


@dataclass
class TokenAnalysis:
    """
    Detailed token analysis result for conversation history.

    Provides comprehensive token statistics including totals, per-role breakdown,
    per-message details, and per-round analysis.

    Attributes:
        total_tokens: Total number of tokens across all messages.
        system_tokens: Number of tokens in system message (if present).
        user_tokens: Total tokens in all user messages.
        assistant_tokens: Total tokens in all assistant messages.
        total_messages: Total number of messages analyzed.
        system_messages: Number of system messages (0 or 1).
        user_messages: Number of user messages.
        assistant_messages: Number of assistant messages.
        per_message: List of (role, content_preview, tokens) tuples for each message.
        per_round: List of (round_index, round_tokens, user_tokens, assistant_tokens) tuples.
        average_tokens_per_message: Average tokens per message.
        average_tokens_per_round: Average tokens per round.
        max_message_tokens: Maximum tokens in a single message.
        min_message_tokens: Minimum tokens in a single message.
        token_distribution: Dict mapping role to total tokens for that role.

    Examples:
        >>> from lexilux import ChatHistory, Tokenizer
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
        >>> history = ChatHistory.from_messages("Hello")
        >>> analysis = history.analyze_tokens(tokenizer)
        >>> print(f"Total tokens: {analysis.total_tokens}")
        >>> print(f"User tokens: {analysis.user_tokens}")
        >>> print(f"Assistant tokens: {analysis.assistant_tokens}")
    """

    total_tokens: int
    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    total_messages: int
    system_messages: int
    user_messages: int
    assistant_messages: int
    per_message: list[tuple[str, str, int]]  # (role, content_preview, tokens)
    per_round: list[tuple[int, int, int, int]]  # (round_index, total, user, assistant)
    average_tokens_per_message: float
    average_tokens_per_round: float
    max_message_tokens: int
    min_message_tokens: int
    token_distribution: dict[str, int]

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TokenAnalysis(total={self.total_tokens}, "
            f"user={self.user_tokens}, assistant={self.assistant_tokens}, "
            f"rounds={len(self.per_round)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "user_tokens": self.user_tokens,
            "assistant_tokens": self.assistant_tokens,
            "total_messages": self.total_messages,
            "system_messages": self.system_messages,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "per_message": [
                {"role": role, "content_preview": preview, "tokens": tokens}
                for role, preview, tokens in self.per_message
            ],
            "per_round": [
                {
                    "round_index": idx,
                    "total_tokens": total,
                    "user_tokens": user,
                    "assistant_tokens": assistant,
                }
                for idx, total, user, assistant in self.per_round
            ],
            "average_tokens_per_message": self.average_tokens_per_message,
            "average_tokens_per_round": self.average_tokens_per_round,
            "max_message_tokens": self.max_message_tokens,
            "min_message_tokens": self.min_message_tokens,
            "token_distribution": self.token_distribution,
        }


class ChatHistory(MutableSequence):
    """
    Conversation history manager.

    Implements MutableSequence protocol, allowing array-like operations:
    - Index access: history[0]
    - Slicing: history[1:5] (returns new ChatHistory)
    - Iteration: for msg in history
    - Length: len(history)
    - Membership: msg in history

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

        # Array-like operations
        >>> msg = history[0]  # Get first message
        >>> first_3 = history[:3]  # Get first 3 messages (new ChatHistory)
        >>> for msg in history:  # Iterate
        ...     print(msg)
        >>> len(history)  # Get length
        >>> msg in history  # Check membership
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

    def add_system(self, content: str) -> None:
        """Add system message (updates system attribute)."""
        self.system = content

    def remove_last(self) -> dict[str, str] | None:
        """
        Remove and return the last message.

        Returns:
            The removed message dict, or None if history is empty.
        """
        if not self.messages:
            return None
        return self.messages.pop()

    def remove_at(self, index: int) -> dict[str, str] | None:
        """
        Remove and return message at specified index.

        Args:
            index: Index of message to remove.

        Returns:
            The removed message dict, or None if index is out of range.
        """
        if 0 <= index < len(self.messages):
            return self.messages.pop(index)
        return None

    def replace_at(self, index: int, role: str, content: str) -> None:
        """
        Replace message at specified index.

        Args:
            index: Index of message to replace.
            role: New role.
            content: New content.

        Raises:
            IndexError: If index is out of range.
        """
        if not (0 <= index < len(self.messages)):
            raise IndexError(f"Index {index} out of range for history with {len(self.messages)} messages")
        self.messages[index] = {"role": role, "content": content}

    def get_user_messages(self) -> list[str]:
        """
        Get all user messages.

        Returns:
            List of user message contents.
        """
        return [msg["content"] for msg in self.messages if msg.get("role") == "user"]

    def get_assistant_messages(self) -> list[str]:
        """
        Get all assistant messages.

        Returns:
            List of assistant message contents.
        """
        return [msg["content"] for msg in self.messages if msg.get("role") == "assistant"]

    def get_last_message(self) -> dict[str, str] | None:
        """
        Get the last message.

        Returns:
            Last message dict, or None if history is empty.
        """
        return self.messages[-1] if self.messages else None

    def get_last_user_message(self) -> str | None:
        """
        Get the last user message content.

        Returns:
            Last user message content, or None if no user messages exist.
        """
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def clone(self) -> ChatHistory:
        """
        Create a deep copy of this history.

        Returns:
            New ChatHistory instance with copied messages.
        """
        return ChatHistory(
            messages=[msg.copy() for msg in self.messages],
            system=self.system,
        )

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

        This is a convenience method that returns only the total token count.
        For detailed analysis, use :meth:`analyze_tokens` instead.

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            Total token count across all messages (including system message).

        Examples:
            >>> from lexilux import ChatHistory, Tokenizer
            >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
            >>> history = ChatHistory.from_messages("Hello")
            >>> total = history.count_tokens(tokenizer)
            >>> print(f"Total tokens: {total}")

        See Also:
            :meth:`analyze_tokens` - For detailed token analysis
        """
        messages = self.get_messages(include_system=True)
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            result = tokenizer(content)
            total += result.usage.total_tokens or 0
        return total

    def count_tokens_per_round(self, tokenizer: Tokenizer) -> list[tuple[int, int]]:
        """
        Count tokens per round.

        This method returns a simple list of (round_index, total_tokens) tuples.
        For more detailed per-round analysis (including user/assistant breakdown),
        use :meth:`analyze_tokens` instead.

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            List of (round_index, total_tokens) tuples, where round_index is 0-based.

        Examples:
            >>> from lexilux import ChatHistory, Tokenizer
            >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
            >>> history = ChatHistory.from_messages("Hello")
            >>> history.add_assistant("Hi!")
            >>> round_tokens = history.count_tokens_per_round(tokenizer)
            >>> for idx, tokens in round_tokens:
            ...     print(f"Round {idx}: {tokens} tokens")

        See Also:
            :meth:`analyze_tokens` - For detailed per-round analysis with role breakdown
        """
        rounds = self._get_rounds()
        result = []
        for idx, round_messages in enumerate(rounds):
            round_tokens = 0
            for msg in round_messages:
                content = msg.get("content", "")
                token_result = tokenizer(content)
                round_tokens += token_result.usage.total_tokens or 0
            result.append((idx, round_tokens))
        return result

    def count_tokens_by_role(self, tokenizer: Tokenizer) -> dict[str, int]:
        """
        Count tokens grouped by role (system, user, assistant).

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            Dictionary mapping role to total token count for that role.
            Keys: "system", "user", "assistant"

        Examples:
            >>> from lexilux import ChatHistory, Tokenizer
            >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
            >>> history = ChatHistory(system="You are helpful")
            >>> history.add_user("Hello")
            >>> history.add_assistant("Hi!")
            >>> role_tokens = history.count_tokens_by_role(tokenizer)
            >>> print(f"User tokens: {role_tokens['user']}")
            >>> print(f"Assistant tokens: {role_tokens['assistant']}")
        """
        messages = self.get_messages(include_system=True)
        role_tokens: dict[str, int] = {"system": 0, "user": 0, "assistant": 0}

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            result = tokenizer(content)
            tokens = result.usage.total_tokens or 0
            if role in role_tokens:
                role_tokens[role] += tokens

        return role_tokens

    def analyze_tokens(self, tokenizer: Tokenizer) -> TokenAnalysis:
        """
        Perform comprehensive token analysis on conversation history.

        This method provides detailed token statistics including:
        - Total tokens and breakdown by role
        - Per-message token counts with content previews
        - Per-round analysis with user/assistant breakdown
        - Statistical metrics (averages, min, max)
        - Token distribution by role

        Args:
            tokenizer: Tokenizer instance.

        Returns:
            TokenAnalysis object containing comprehensive token statistics.

        Examples:
            Basic usage:
            >>> from lexilux import ChatHistory, Tokenizer
            >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
            >>> history = ChatHistory(system="You are helpful")
            >>> history.add_user("What is Python?")
            >>> history.add_assistant("Python is a programming language.")
            >>> analysis = history.analyze_tokens(tokenizer)
            >>> print(f"Total: {analysis.total_tokens}")
            >>> print(f"User: {analysis.user_tokens}, Assistant: {analysis.assistant_tokens}")

            Detailed analysis:
            >>> analysis = history.analyze_tokens(tokenizer)
            >>> # Per-message breakdown
            >>> for role, preview, tokens in analysis.per_message:
            ...     print(f"{role}: {preview[:30]}... ({tokens} tokens)")
            >>> # Per-round breakdown
            >>> for idx, total, user, assistant in analysis.per_round:
            ...     print(f"Round {idx}: total={total}, user={user}, assistant={assistant}")
            >>> # Distribution
            >>> print(f"Distribution: {analysis.token_distribution}")

            Export analysis:
            >>> analysis_dict = analysis.to_dict()
            >>> import json
            >>> print(json.dumps(analysis_dict, indent=2))
        """
        messages = self.get_messages(include_system=True)
        rounds = self._get_rounds()

        # Initialize counters
        total_tokens = 0
        system_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        system_count = 0
        user_count = 0
        assistant_count = 0

        # Per-message analysis
        per_message: list[tuple[str, str, int]] = []
        message_tokens_list: list[int] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            result = tokenizer(content)
            tokens = result.usage.total_tokens or 0

            total_tokens += tokens
            message_tokens_list.append(tokens)

            # Content preview (first 50 chars)
            preview = content[:50] + "..." if len(content) > 50 else content
            per_message.append((role, preview, tokens))

            # Count by role
            if role == "system":
                system_tokens += tokens
                system_count += 1
            elif role == "user":
                user_tokens += tokens
                user_count += 1
            elif role == "assistant":
                assistant_tokens += tokens
                assistant_count += 1

        # Per-round analysis
        per_round: list[tuple[int, int, int, int]] = []
        round_tokens_list: list[int] = []

        for idx, round_messages in enumerate(rounds):
            round_total = 0
            round_user = 0
            round_assistant = 0

            for msg in round_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                result = tokenizer(content)
                tokens = result.usage.total_tokens or 0

                round_total += tokens
                if role == "user":
                    round_user += tokens
                elif role == "assistant":
                    round_assistant += tokens

            per_round.append((idx, round_total, round_user, round_assistant))
            round_tokens_list.append(round_total)

        # Calculate statistics
        avg_per_message = total_tokens / len(messages) if messages else 0.0
        avg_per_round = (
            sum(round_tokens_list) / len(round_tokens_list) if round_tokens_list else 0.0
        )
        max_message = max(message_tokens_list) if message_tokens_list else 0
        min_message = min(message_tokens_list) if message_tokens_list else 0

        # Token distribution
        token_distribution = {
            "system": system_tokens,
            "user": user_tokens,
            "assistant": assistant_tokens,
        }

        return TokenAnalysis(
            total_tokens=total_tokens,
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens,
            total_messages=len(messages),
            system_messages=system_count,
            user_messages=user_count,
            assistant_messages=assistant_count,
            per_message=per_message,
            per_round=per_round,
            average_tokens_per_message=round(avg_per_message, 2),
            average_tokens_per_round=round(avg_per_round, 2),
            max_message_tokens=max_message,
            min_message_tokens=min_message,
            token_distribution=token_distribution,
        )

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
            system_tokens = sys_result.usage.total_tokens or 0

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

    # MutableSequence protocol implementation
    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __getitem__(self, key: int | slice) -> dict[str, str] | ChatHistory:
        """
        Get message(s) by index or slice.

        Args:
            key: Index (int) or slice.

        Returns:
            Single message dict (index) or new ChatHistory instance (slice).

        Examples:
            >>> history[0]  # Get first message
            >>> history[1:3]  # Get messages at index 1-2, returns new ChatHistory
            >>> history[:5]  # Get first 5 messages
            >>> history[-3:]  # Get last 3 messages
        """
        if isinstance(key, int):
            return self.messages[key]
        elif isinstance(key, slice):
            return ChatHistory(
                messages=self.messages[key].copy(),
                system=self.system,
            )
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key: int | slice, value: dict[str, str] | Sequence[dict[str, str]]) -> None:
        """
        Set message(s) by index or slice.

        Args:
            key: Index (int) or slice.
            value: Single message dict (index) or sequence of message dicts (slice).

        Raises:
            TypeError: If value type is invalid.
        """
        if isinstance(key, int):
            if not isinstance(value, dict):
                raise TypeError("Value must be a dict")
            self.messages[key] = value
        elif isinstance(key, slice):
            if not isinstance(value, (list, tuple)):
                raise TypeError("Value must be a list or tuple of dicts")
            self.messages[key] = list(value)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __delitem__(self, key: int | slice) -> None:
        """
        Delete message(s) by index or slice.

        Args:
            key: Index (int) or slice.
        """
        del self.messages[key]

    def insert(self, index: int, value: dict[str, str]) -> None:
        """
        Insert message at specified index.

        Args:
            index: Index to insert at.
            value: Message dict to insert.

        Raises:
            TypeError: If value is not a dict.
        """
        if not isinstance(value, dict):
            raise TypeError("Value must be a dict")
        self.messages.insert(index, value)

    def __iter__(self):
        """Iterate over messages."""
        return iter(self.messages)

    def __contains__(self, item) -> bool:
        """Check if message is in history."""
        return item in self.messages

    def __add__(self, other: ChatHistory) -> ChatHistory:
        """
        Merge two histories (concatenate messages).

        Args:
            other: Another ChatHistory instance.

        Returns:
            New ChatHistory instance with merged messages.
            System message from self is used.

        Examples:
            >>> history1 = ChatHistory.from_messages("Hello")
            >>> history2 = ChatHistory.from_messages("How are you?")
            >>> combined = history1 + history2
        """
        return ChatHistory(
            messages=self.messages + other.messages,
            system=self.system,  # Use self's system message
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ChatHistory(messages={len(self.messages)}, system={self.system is not None})"

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


# Utility functions for ChatHistory operations


def merge_histories(*histories: ChatHistory) -> ChatHistory:
    """
    Merge multiple conversation histories into one.

    Args:
        *histories: Multiple ChatHistory instances to merge.

    Returns:
        New ChatHistory instance with merged messages.

    Examples:
        >>> history1 = ChatHistory.from_messages("Hello")
        >>> history2 = ChatHistory.from_messages("How are you?")
        >>> merged = merge_histories(history1, history2)
    """
    if not histories:
        return ChatHistory()

    # Use first history's system message (if any)
    system = histories[0].system

    # Merge all messages
    merged_messages = []
    for history in histories:
        merged_messages.extend(history.messages)

    return ChatHistory(messages=merged_messages, system=system)


def filter_by_role(history: ChatHistory, role: str) -> ChatHistory:
    """
    Filter history by role.

    Args:
        history: ChatHistory instance to filter.
        role: Role to filter by ("user", "assistant", "system", "tool").

    Returns:
        New ChatHistory instance with filtered messages.

    Examples:
        >>> history = ChatHistory.from_messages(["Hello", "Hi there"])
        >>> user_only = filter_by_role(history, "user")
    """
    filtered_messages = [msg for msg in history.messages if msg.get("role") == role]

    # Include system message if filtering by system role
    system = history.system if role == "system" else None

    return ChatHistory(messages=filtered_messages, system=system)


def search_content(history: ChatHistory, pattern: str) -> list[dict[str, str]]:
    """
    Search for messages containing the pattern.

    Args:
        history: ChatHistory instance to search.
        pattern: Search pattern (case-sensitive substring match).

    Returns:
        List of message dictionaries that contain the pattern.

    Examples:
        >>> history = ChatHistory.from_messages(["Hello world", "Hi there"])
        >>> results = search_content(history, "world")
        >>> # Returns messages containing "world"
    """
    results = []
    for msg in history.get_messages(include_system=True):
        content = msg.get("content", "")
        if pattern in content:
            results.append(msg)
    return results


def get_statistics(history: ChatHistory, tokenizer: Tokenizer | None = None) -> dict[str, Any]:
    """
    Get comprehensive statistics about the conversation history.

    This function provides both character-based and token-based statistics.
    If a tokenizer is provided, token statistics are included.

    Args:
        history: ChatHistory instance to analyze.
        tokenizer: Optional Tokenizer instance for token statistics.
                  If provided, includes detailed token analysis.

    Returns:
        Dictionary with statistics:
        - total_rounds: Number of conversation rounds
        - total_messages: Total number of messages
        - user_messages: Number of user messages
        - assistant_messages: Number of assistant messages
        - system_messages: Number of system messages (0 or 1)
        - has_system: Whether system message exists
        - total_characters: Total characters across all messages
        - average_message_length: Average length of messages (characters)
        - token_analysis: TokenAnalysis object (if tokenizer provided)
        - total_tokens: Total tokens (if tokenizer provided)
        - tokens_by_role: Dict of tokens by role (if tokenizer provided)

    Examples:
        Basic statistics (character-based):
        >>> history = ChatHistory.from_messages(["Hello", "Hi"])
        >>> stats = get_statistics(history)
        >>> print(stats["total_rounds"])

        With token analysis:
        >>> from lexilux import Tokenizer
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
        >>> stats = get_statistics(history, tokenizer=tokenizer)
        >>> print(f"Total tokens: {stats['total_tokens']}")
        >>> print(f"User tokens: {stats['tokens_by_role']['user']}")
    """
    messages = history.get_messages(include_system=True)
    rounds = history._get_rounds()

    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    system_count = 1 if history.system else 0

    total_length = sum(len(msg.get("content", "")) for msg in messages)
    avg_length = total_length / len(messages) if messages else 0

    stats: dict[str, Any] = {
        "total_rounds": len(rounds),
        "total_messages": len(messages),
        "user_messages": user_count,
        "assistant_messages": assistant_count,
        "system_messages": system_count,
        "has_system": history.system is not None,
        "total_characters": total_length,
        "average_message_length": round(avg_length, 2),
    }

    # Add token statistics if tokenizer provided
    if tokenizer is not None:
        analysis = history.analyze_tokens(tokenizer)
        stats["token_analysis"] = analysis
        stats["total_tokens"] = analysis.total_tokens
        stats["tokens_by_role"] = analysis.token_distribution
        stats["average_tokens_per_message"] = analysis.average_tokens_per_message
        stats["average_tokens_per_round"] = analysis.average_tokens_per_round
        stats["max_message_tokens"] = analysis.max_message_tokens
        stats["min_message_tokens"] = analysis.min_message_tokens

    return stats
