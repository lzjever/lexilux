"""
Tests for reasoning mode support.

Tests the unified reasoning API across providers:
- normalize_reasoning()
- build_reasoning_request()
- Provider-specific reasoning configurations
- ChatResult.reasoning and ChatStreamChunk.reasoning
"""

import pytest

from lexilux.chat.reasoning import (
    build_reasoning_request,
    extract_reasoning_content,
    extract_streaming_reasoning_delta,
    normalize_reasoning,
)
from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.providers import ReasoningConfig, get_reasoning_config
from lexilux.usage import Usage


class TestNormalizeReasoning:
    """Tests for normalize_reasoning function."""

    def test_none_returns_empty(self):
        assert normalize_reasoning(None) == {}

    def test_false_returns_empty(self):
        assert normalize_reasoning(False) == {}

    def test_true_returns_enabled(self):
        assert normalize_reasoning(True) == {"enabled": True}

    def test_dict_passthrough(self):
        result = normalize_reasoning({"effort": "high", "max_tokens": 10000})
        assert result == {"enabled": True, "effort": "high", "max_tokens": 10000}


class TestBuildReasoningRequest:
    """Tests for build_reasoning_request function."""

    def test_disabled_returns_empty(self):
        result = build_reasoning_request("deepseek", {"enabled": False})
        assert result == {}

    def test_unknown_provider_returns_empty(self):
        result = build_reasoning_request("unknown_provider", {"enabled": True})
        assert result == {}

    def test_deepseek_extra_body(self):
        result = build_reasoning_request("deepseek", {"enabled": True})
        assert result == {"thinking": {"type": "enabled"}}

    def test_openai_reasoning_param(self):
        result = build_reasoning_request("openai", {"enabled": True})
        assert result == {"reasoning": {"effort": "medium"}}

    def test_openai_custom_effort(self):
        result = build_reasoning_request("openai", {"enabled": True, "effort": "high"})
        assert result == {"reasoning": {"effort": "high"}}

    def test_anthropic_thinking_param(self):
        result = build_reasoning_request("anthropic", {"enabled": True})
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        # budget_tokens only added when effort or max_tokens is specified

    def test_anthropic_custom_budget(self):
        result = build_reasoning_request(
            "anthropic", {"enabled": True, "max_tokens": 20000}
        )
        assert result["thinking"]["budget_tokens"] == 20000

    def test_anthropic_effort_to_budget(self):
        result = build_reasoning_request(
            "anthropic", {"enabled": True, "effort": "high"}
        )
        assert result["thinking"]["budget_tokens"] == 32768

    def test_kimi_model_selection(self):
        # Kimi uses model selection, no extra params
        result = build_reasoning_request("moonshotai", {"enabled": True})
        assert result == {}

    def test_zhipu_extra_body(self):
        result = build_reasoning_request("zhipuai", {"enabled": True})
        assert result == {"thinking": {"type": "enabled"}}


class TestProviderReasoningConfigs:
    """Tests for provider reasoning configurations."""

    def test_deepseek_config(self):
        config = get_reasoning_config("deepseek")
        assert config is not None
        assert config.method == "extra_body"
        assert config.response_field == "reasoning_content"

    def test_openai_config(self):
        config = get_reasoning_config("openai")
        assert config is not None
        assert config.method == "reasoning_param"
        assert config.response_field is None  # Hidden

    def test_anthropic_config(self):
        config = get_reasoning_config("anthropic")
        assert config is not None
        assert config.method == "thinking_param"
        assert config.response_field == "thinking"
        assert config.supports_budget is True

    def test_unknown_provider_returns_none(self):
        config = get_reasoning_config("nonexistent_provider")
        assert config is None


class TestExtractReasoningContent:
    """Tests for extract_reasoning_content function."""

    def test_extracts_reasoning_content_field(self):
        response = {
            "choices": [
                {"message": {"content": "Answer", "reasoning_content": "My reasoning"}}
            ]
        }
        result = extract_reasoning_content(response, "deepseek")
        assert result == "My reasoning"

    def test_no_reasoning_field_returns_none(self):
        response = {"choices": [{"message": {"content": "Answer"}}]}
        result = extract_reasoning_content(response, "deepseek")
        assert result is None

    def test_openai_returns_none(self):
        # OpenAI reasoning is hidden
        response = {"choices": [{"message": {"content": "Answer"}}]}
        result = extract_reasoning_content(response, "openai")
        assert result is None

    def test_anthropic_content_blocks(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "Step 1..."},
                            {"type": "thinking", "thinking": "Step 2..."},
                            {"type": "text", "text": "Answer"},
                        ]
                    }
                }
            ]
        }
        result = extract_reasoning_content(response, "anthropic")
        assert "Step 1..." in result
        assert "Step 2..." in result


class TestExtractStreamingReasoningDelta:
    """Tests for extract_streaming_reasoning_delta function."""

    def test_extracts_reasoning_delta(self):
        chunk = {
            "choices": [{"delta": {"reasoning_content": "thinking more"}}]
        }
        result = extract_streaming_reasoning_delta(chunk, "deepseek")
        assert result == "thinking more"

    def test_no_delta_returns_empty(self):
        chunk = {"choices": [{"delta": {"content": "text"}}]}
        result = extract_streaming_reasoning_delta(chunk, "deepseek")
        assert result == ""


class TestChatResultReasoning:
    """Tests for ChatResult reasoning field."""

    def test_reasoning_field(self):
        result = ChatResult(
            text="The answer is 42",
            usage=Usage(input_tokens=10, output_tokens=20),
            reasoning="Let me think about this...",
        )
        assert result.reasoning == "Let me think about this..."

    def test_has_reasoning_true(self):
        result = ChatResult(
            text="Answer",
            usage=Usage(input_tokens=10, output_tokens=20),
            reasoning="Reasoning here",
        )
        assert result.has_reasoning is True

    def test_has_reasoning_false(self):
        result = ChatResult(
            text="Answer",
            usage=Usage(input_tokens=10, output_tokens=20),
        )
        assert result.has_reasoning is False

    def test_empty_reasoning_has_reasoning_false(self):
        result = ChatResult(
            text="Answer",
            usage=Usage(input_tokens=10, output_tokens=20),
            reasoning="",
        )
        assert result.has_reasoning is False


class TestChatStreamChunkReasoning:
    """Tests for ChatStreamChunk reasoning properties."""

    def test_reasoning_property(self):
        chunk = ChatStreamChunk(
            delta="",
            usage=Usage(input_tokens=0, output_tokens=0),
            done=False,
            reasoning_content="thinking step 1",
        )
        assert chunk.reasoning == "thinking step 1"

    def test_has_reasoning_true(self):
        chunk = ChatStreamChunk(
            delta="",
            usage=Usage(input_tokens=0, output_tokens=0),
            done=False,
            reasoning_content="thinking",
        )
        assert chunk.has_reasoning is True

    def test_has_reasoning_false(self):
        chunk = ChatStreamChunk(
            delta="text",
            usage=Usage(input_tokens=0, output_tokens=0),
            done=False,
        )
        assert chunk.has_reasoning is False

    def test_reasoning_defaults_to_empty(self):
        chunk = ChatStreamChunk(
            delta="text",
            usage=Usage(input_tokens=0, output_tokens=0),
            done=False,
        )
        assert chunk.reasoning == ""


class TestReasoningConfigDataclass:
    """Tests for ReasoningConfig dataclass."""

    def test_basic_config(self):
        config = ReasoningConfig(
            method="extra_body",
            response_field="reasoning_content",
        )
        assert config.method == "extra_body"
        assert config.response_field == "reasoning_content"
        assert config.default_effort == "medium"

    def test_full_config(self):
        config = ReasoningConfig(
            method="thinking_param",
            response_field="thinking",
            default_effort="high",
            supports_budget=True,
            effort_to_budget={"low": 1024, "high": 32768},
            params={"type": "enabled"},
        )
        assert config.method == "thinking_param"
        assert config.supports_budget is True
        assert config.effort_to_budget["high"] == 32768
