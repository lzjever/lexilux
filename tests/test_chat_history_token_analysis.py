"""
Comprehensive tests for ChatHistory token analysis functionality.

Tests cover:
- TokenAnalysis dataclass
- analyze_tokens() method
- count_tokens_by_role() method
- Enhanced get_statistics() with tokenizer
- Integration with truncation
"""

from lexilux import ChatHistory, TokenAnalysis
from lexilux.chat import get_statistics


class MockTokenizer:
    """
    Mock tokenizer for testing that counts words as tokens.

    This mock follows the TokenizeResult interface:
    - TokenizeResult(input_ids, attention_mask, usage, raw)
    - usage.total_tokens contains the token count
    """

    def __init__(self, model: str = "test-model"):
        self.model = model

    def __call__(self, text: str):
        """
        Count words as tokens for testing.

        Returns TokenizeResult following the interface:
        - input_ids: list[list[int]] - token IDs
        - attention_mask: list[list[int]] | None
        - usage: Usage with total_tokens
        - raw: dict | None
        """
        from lexilux import TokenizeResult, Usage

        words = text.split()
        token_count = len(words) if words else 0

        # Create usage with total_tokens
        usage = Usage(
            input_tokens=token_count,
            output_tokens=None,
            total_tokens=token_count,
        )

        # Create input_ids (list of token ID sequences)
        # For simplicity, use word index as token ID
        input_ids = [[i for i in range(token_count)]] if token_count > 0 else [[]]

        return TokenizeResult(
            input_ids=input_ids,
            attention_mask=None,
            usage=usage,
            raw=None,
        )


class TestTokenAnalysisDataclass:
    """Test TokenAnalysis dataclass structure and methods"""

    def test_token_analysis_init(self):
        """Test TokenAnalysis initialization"""
        analysis = TokenAnalysis(
            total_tokens=100,
            system_tokens=10,
            user_tokens=30,
            assistant_tokens=60,
            total_messages=5,
            system_messages=1,
            user_messages=2,
            assistant_messages=2,
            per_message=[("system", "preview", 10), ("user", "preview", 15)],
            per_round=[(0, 45, 15, 30)],
            average_tokens_per_message=20.0,
            average_tokens_per_round=45.0,
            max_message_tokens=30,
            min_message_tokens=10,
            token_distribution={"system": 10, "user": 30, "assistant": 60},
        )
        assert analysis.total_tokens == 100
        assert analysis.system_tokens == 10
        assert analysis.user_tokens == 30
        assert analysis.assistant_tokens == 60

    def test_token_analysis_repr(self):
        """Test TokenAnalysis string representation"""
        analysis = TokenAnalysis(
            total_tokens=100,
            system_tokens=10,
            user_tokens=30,
            assistant_tokens=60,
            total_messages=3,
            system_messages=1,
            user_messages=1,
            assistant_messages=1,
            per_message=[],
            per_round=[],
            average_tokens_per_message=33.33,
            average_tokens_per_round=50.0,
            max_message_tokens=30,
            min_message_tokens=10,
            token_distribution={"system": 10, "user": 30, "assistant": 60},
        )
        repr_str = repr(analysis)
        assert "TokenAnalysis" in repr_str
        assert "total=100" in repr_str
        assert "user=30" in repr_str
        assert "assistant=60" in repr_str

    def test_token_analysis_to_dict(self):
        """Test TokenAnalysis.to_dict() method"""
        analysis = TokenAnalysis(
            total_tokens=100,
            system_tokens=10,
            user_tokens=30,
            assistant_tokens=60,
            total_messages=3,
            system_messages=1,
            user_messages=1,
            assistant_messages=1,
            per_message=[("system", "preview", 10)],
            per_round=[(0, 45, 15, 30)],
            average_tokens_per_message=33.33,
            average_tokens_per_round=45.0,
            max_message_tokens=30,
            min_message_tokens=10,
            token_distribution={"system": 10, "user": 30, "assistant": 60},
        )
        data = analysis.to_dict()
        assert isinstance(data, dict)
        assert data["total_tokens"] == 100
        assert data["system_tokens"] == 10
        assert data["user_tokens"] == 30
        assert data["assistant_tokens"] == 60
        assert isinstance(data["per_message"], list)
        assert isinstance(data["per_round"], list)
        assert isinstance(data["token_distribution"], dict)
        assert "average_tokens_per_message" in data
        assert "average_tokens_per_round" in data


class TestCountTokensByRole:
    """Test count_tokens_by_role() method"""

    def test_count_tokens_by_role_basic(self):
        """Test basic role-based token counting"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello world")
        history.add_assistant("Hi there friend")

        role_tokens = history.count_tokens_by_role(tokenizer)
        assert isinstance(role_tokens, dict)
        assert "system" in role_tokens
        assert "user" in role_tokens
        assert "assistant" in role_tokens
        assert role_tokens["system"] == 3  # "You are helpful"
        assert role_tokens["user"] == 2  # "Hello world"
        assert role_tokens["assistant"] == 3  # "Hi there friend"

    def test_count_tokens_by_role_no_system(self):
        """Test role-based counting without system message"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi")

        role_tokens = history.count_tokens_by_role(tokenizer)
        assert role_tokens["system"] == 0
        assert role_tokens["user"] == 1
        assert role_tokens["assistant"] == 1

    def test_count_tokens_by_role_multiple_messages(self):
        """Test role-based counting with multiple messages per role"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("First question")
        history.add_assistant("First answer")
        history.add_user("Second question")
        history.add_assistant("Second answer")

        role_tokens = history.count_tokens_by_role(tokenizer)
        assert role_tokens["user"] == 4  # 2 + 2
        assert role_tokens["assistant"] == 4  # 2 + 2

    def test_count_tokens_by_role_empty_history(self):
        """Test role-based counting with empty history"""
        tokenizer = MockTokenizer()
        history = ChatHistory()

        role_tokens = history.count_tokens_by_role(tokenizer)
        assert role_tokens["system"] == 0
        assert role_tokens["user"] == 0
        assert role_tokens["assistant"] == 0


class TestAnalyzeTokens:
    """Test analyze_tokens() method"""

    def test_analyze_tokens_basic(self):
        """Test basic token analysis"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("What is Python?")
        history.add_assistant("Python is a programming language")

        analysis = history.analyze_tokens(tokenizer)
        assert isinstance(analysis, TokenAnalysis)
        assert analysis.total_tokens > 0
        assert analysis.system_tokens > 0
        assert analysis.user_tokens > 0
        assert analysis.assistant_tokens > 0

    def test_analyze_tokens_summary_statistics(self):
        """Test summary statistics in analysis"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        analysis = history.analyze_tokens(tokenizer)
        # System: 3 tokens, User: 1 token, Assistant: 2 tokens
        assert analysis.total_tokens == 6  # 3 + 1 + 2
        assert analysis.system_tokens == 3
        assert analysis.user_tokens == 1
        assert analysis.assistant_tokens == 2
        assert analysis.total_messages == 3
        assert analysis.system_messages == 1
        assert analysis.user_messages == 1
        assert analysis.assistant_messages == 1

    def test_analyze_tokens_per_message(self):
        """Test per-message breakdown"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        analysis = history.analyze_tokens(tokenizer)
        assert len(analysis.per_message) == 3
        assert all(
            isinstance(item, tuple) and len(item) == 3 for item in analysis.per_message
        )
        # Check structure: (role, preview, tokens)
        for role, preview, tokens in analysis.per_message:
            assert role in ["system", "user", "assistant"]
            assert isinstance(preview, str)
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_analyze_tokens_per_round(self):
        """Test per-round breakdown"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("First question")
        history.add_assistant("First answer")
        history.add_user("Second question")
        history.add_assistant("Second answer")

        analysis = history.analyze_tokens(tokenizer)
        assert len(analysis.per_round) == 2  # Two rounds
        # Check structure: (round_index, total, user, assistant)
        for idx, total, user, assistant in analysis.per_round:
            assert isinstance(idx, int)
            assert isinstance(total, int)
            assert isinstance(user, int)
            assert isinstance(assistant, int)
            assert total == user + assistant
            assert total > 0

    def test_analyze_tokens_statistical_metrics(self):
        """Test statistical metrics"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("Short")
        history.add_assistant("This is a longer response with more words")
        history.add_user("Another short")
        history.add_assistant("Another longer response")

        analysis = history.analyze_tokens(tokenizer)
        assert analysis.average_tokens_per_message > 0
        assert analysis.average_tokens_per_round > 0
        assert analysis.max_message_tokens > 0
        assert analysis.min_message_tokens > 0
        assert analysis.max_message_tokens >= analysis.min_message_tokens
        assert analysis.average_tokens_per_message >= analysis.min_message_tokens
        assert analysis.average_tokens_per_message <= analysis.max_message_tokens

    def test_analyze_tokens_distribution(self):
        """Test token distribution by role"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        analysis = history.analyze_tokens(tokenizer)
        assert isinstance(analysis.token_distribution, dict)
        assert "system" in analysis.token_distribution
        assert "user" in analysis.token_distribution
        assert "assistant" in analysis.token_distribution
        assert (
            analysis.token_distribution["system"]
            + analysis.token_distribution["user"]
            + analysis.token_distribution["assistant"]
            == analysis.total_tokens
        )

    def test_analyze_tokens_empty_history(self):
        """Test analysis with empty history"""
        tokenizer = MockTokenizer()
        history = ChatHistory()

        analysis = history.analyze_tokens(tokenizer)
        assert analysis.total_tokens == 0
        assert analysis.system_tokens == 0
        assert analysis.user_tokens == 0
        assert analysis.assistant_tokens == 0
        assert len(analysis.per_message) == 0
        assert len(analysis.per_round) == 0

    def test_analyze_tokens_only_system(self):
        """Test analysis with only system message"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")

        analysis = history.analyze_tokens(tokenizer)
        assert analysis.total_tokens > 0
        assert analysis.system_tokens > 0
        assert analysis.user_tokens == 0
        assert analysis.assistant_tokens == 0
        assert analysis.system_messages == 1

    def test_analyze_tokens_incomplete_round(self):
        """Test analysis with incomplete round (user without assistant)"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("Question without answer")

        analysis = history.analyze_tokens(tokenizer)
        assert analysis.total_tokens > 0
        assert analysis.user_tokens > 0
        assert analysis.assistant_tokens == 0
        # Incomplete round should still be counted
        assert len(analysis.per_round) == 1

    def test_analyze_tokens_multiple_rounds(self):
        """Test analysis with multiple conversation rounds"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        for i in range(3):
            history.add_user(f"Question {i}")
            history.add_assistant(f"Answer {i}")

        analysis = history.analyze_tokens(tokenizer)
        assert len(analysis.per_round) == 3
        assert analysis.user_messages == 3
        assert analysis.assistant_messages == 3
        # Each round should have tokens
        for idx, total, user, assistant in analysis.per_round:
            assert total > 0
            assert user > 0
            assert assistant > 0


class TestGetStatisticsWithTokenizer:
    """Test enhanced get_statistics() with tokenizer parameter"""

    def test_get_statistics_without_tokenizer(self):
        """Test get_statistics without tokenizer (character-based only)"""
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi")

        stats = get_statistics(history)
        assert "total_rounds" in stats
        assert "total_messages" in stats
        assert "total_characters" in stats
        # Token-related fields should not be present
        assert "total_tokens" not in stats
        assert "tokens_by_role" not in stats
        assert "token_analysis" not in stats

    def test_get_statistics_with_tokenizer(self):
        """Test get_statistics with tokenizer (includes token analysis)"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        stats = get_statistics(history, tokenizer=tokenizer)
        # Character-based stats
        assert "total_rounds" in stats
        assert "total_messages" in stats
        assert "total_characters" in stats
        # Token-based stats
        assert "total_tokens" in stats
        assert "tokens_by_role" in stats
        assert "token_analysis" in stats
        assert isinstance(stats["token_analysis"], TokenAnalysis)

    def test_get_statistics_token_values(self):
        """Test token values in statistics"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        stats = get_statistics(history, tokenizer=tokenizer)
        assert stats["total_tokens"] > 0
        assert isinstance(stats["tokens_by_role"], dict)
        assert "system" in stats["tokens_by_role"]
        assert "user" in stats["tokens_by_role"]
        assert "assistant" in stats["tokens_by_role"]
        assert stats["average_tokens_per_message"] > 0
        assert stats["average_tokens_per_round"] > 0
        assert stats["max_message_tokens"] > 0
        assert stats["min_message_tokens"] > 0

    def test_get_statistics_consistency(self):
        """Test consistency between token_analysis and direct stats"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        stats = get_statistics(history, tokenizer=tokenizer)
        analysis = stats["token_analysis"]

        # Values should match
        assert stats["total_tokens"] == analysis.total_tokens
        assert stats["tokens_by_role"] == analysis.token_distribution
        assert (
            stats["average_tokens_per_message"] == analysis.average_tokens_per_message
        )
        assert stats["average_tokens_per_round"] == analysis.average_tokens_per_round
        assert stats["max_message_tokens"] == analysis.max_message_tokens
        assert stats["min_message_tokens"] == analysis.min_message_tokens


class TestTokenAnalysisIntegration:
    """Test integration of token analysis with other features"""

    def test_analyze_tokens_with_truncation(self):
        """Test token analysis before and after truncation"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        # Create multiple rounds
        for i in range(5):
            history.add_user(f"Question {i} with more words")
            history.add_assistant(f"Answer {i} with many words in response")

        # Analyze before truncation
        analysis_before = history.analyze_tokens(tokenizer)
        assert analysis_before.total_tokens > 0
        assert len(analysis_before.per_round) == 5

        # Truncate
        truncated = history.truncate_by_rounds(
            tokenizer, max_tokens=20, keep_system=True
        )

        # Analyze after truncation
        analysis_after = truncated.analyze_tokens(tokenizer)
        assert analysis_after.total_tokens <= 20
        assert analysis_after.total_tokens < analysis_before.total_tokens
        assert len(analysis_after.per_round) <= len(analysis_before.per_round)

    def test_analyze_tokens_round_consistency(self):
        """Test that per_round tokens match per_message tokens"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("First question")
        history.add_assistant("First answer")
        history.add_user("Second question")
        history.add_assistant("Second answer")

        analysis = history.analyze_tokens(tokenizer)

        # Calculate round totals from per_message
        round_totals_from_messages = {}
        current_round = 0
        for role, preview, tokens in analysis.per_message:
            if role == "user":
                if current_round in round_totals_from_messages:
                    current_round += 1
                round_totals_from_messages[current_round] = {"user": 0, "assistant": 0}
                round_totals_from_messages[current_round]["user"] = tokens
            elif role == "assistant":
                round_totals_from_messages[current_round]["assistant"] = tokens

        # Compare with per_round
        for idx, total, user, assistant in analysis.per_round:
            if idx in round_totals_from_messages:
                expected_user = round_totals_from_messages[idx]["user"]
                expected_assistant = round_totals_from_messages[idx]["assistant"]
                assert user == expected_user
                assert assistant == expected_assistant
                assert total == user + assistant

    def test_analyze_tokens_content_preview(self):
        """Test that content previews are generated correctly"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        # Long message
        long_message = (
            "This is a very long message that should be truncated in the preview"
        )
        history.add_user(long_message)

        analysis = history.analyze_tokens(tokenizer)
        user_message = next(
            (msg for msg in analysis.per_message if msg[0] == "user"), None
        )
        assert user_message is not None
        preview = user_message[1]
        # Preview should be truncated (50 chars + "...")
        assert len(preview) <= 53  # 50 + "..."
        assert long_message.startswith(preview.replace("...", ""))

    def test_count_tokens_vs_analyze_tokens(self):
        """Test that count_tokens() matches analyze_tokens().total_tokens"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        total_count = history.count_tokens(tokenizer)
        analysis = history.analyze_tokens(tokenizer)

        assert total_count == analysis.total_tokens

    def test_count_tokens_by_role_vs_analyze_tokens(self):
        """Test that count_tokens_by_role() matches analyze_tokens().token_distribution"""
        tokenizer = MockTokenizer()
        history = ChatHistory(system="You are helpful")
        history.add_user("Hello")
        history.add_assistant("Hi there")

        role_tokens = history.count_tokens_by_role(tokenizer)
        analysis = history.analyze_tokens(tokenizer)

        assert role_tokens["system"] == analysis.token_distribution["system"]
        assert role_tokens["user"] == analysis.token_distribution["user"]
        assert role_tokens["assistant"] == analysis.token_distribution["assistant"]


class TestTokenAnalysisEdgeCases:
    """Test edge cases and error handling"""

    def test_analyze_tokens_empty_strings(self):
        """Test analysis with empty string messages"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("")
        history.add_assistant("")

        analysis = history.analyze_tokens(tokenizer)
        # Empty strings should have 0 tokens
        assert analysis.user_tokens == 0
        assert analysis.assistant_tokens == 0

    def test_analyze_tokens_very_long_message(self):
        """Test analysis with very long message"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        # Create a very long message
        long_content = "word " * 1000  # 1000 words
        history.add_user(long_content)

        analysis = history.analyze_tokens(tokenizer)
        assert analysis.user_tokens == 1000
        assert analysis.max_message_tokens == 1000

    def test_analyze_tokens_special_characters(self):
        """Test analysis with special characters"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("Hello! How are you? I'm fine.")
        history.add_assistant("Great! Let's test: numbers 123, symbols @#$")

        analysis = history.analyze_tokens(tokenizer)
        # Should handle special characters without errors
        assert analysis.total_tokens > 0
        assert all(tokens >= 0 for _, _, tokens in analysis.per_message)

    def test_analyze_tokens_unicode(self):
        """Test analysis with unicode characters"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        history.add_user("你好，世界")
        history.add_assistant("Hello, 世界")

        analysis = history.analyze_tokens(tokenizer)
        # Should handle unicode without errors
        assert analysis.total_tokens > 0

    def test_analyze_tokens_multiline_messages(self):
        """Test analysis with multiline messages"""
        tokenizer = MockTokenizer()
        history = ChatHistory()
        multiline = "Line 1\nLine 2\nLine 3"
        history.add_user(multiline)

        analysis = history.analyze_tokens(tokenizer)
        assert analysis.user_tokens > 0
        # Preview should handle newlines
        user_msg = next((msg for msg in analysis.per_message if msg[0] == "user"), None)
        assert user_msg is not None
