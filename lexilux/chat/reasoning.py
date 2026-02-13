"""
Reasoning mode helper functions.

Provides utilities for normalizing reasoning parameters, building
provider-specific requests, and extracting reasoning content from responses.
"""

from __future__ import annotations

from typing import Any

from lexilux.providers.base import ReasoningConfig
from lexilux.providers.registry import get_reasoning_config


def normalize_reasoning(reasoning: bool | dict | None) -> dict[str, Any]:
    """
    Convert reasoning parameter to normalized dict format.

    Args:
        reasoning: True, False, dict with options, or None.

    Returns:
        Normalized dict with at least {"enabled": bool}.

    Examples:
        >>> normalize_reasoning(True)
        {"enabled": True}
        >>> normalize_reasoning({"effort": "high"})
        {"enabled": True, "effort": "high"}
        >>> normalize_reasoning(False)
        {}
    """
    if reasoning is None or reasoning is False:
        return {}
    if reasoning is True:
        return {"enabled": True}
    return {"enabled": True, **reasoning}


def build_reasoning_request(
    provider_id: str,
    reasoning: dict[str, Any],
) -> dict[str, Any]:
    """
    Build provider-specific request parameters for reasoning.

    Args:
        provider_id: Provider identifier (e.g., "openai", "deepseek").
        reasoning: Normalized reasoning configuration dict.

    Returns:
        Dict with provider-specific parameters to merge into the request payload.
        Empty dict if reasoning is disabled or provider not supported.
    """
    if not reasoning.get("enabled"):
        return {}

    config = get_reasoning_config(provider_id)
    if not config:
        return {}

    if config.method == "extra_body":
        # DeepSeek, Zhipu, Minimax style - merge params directly into payload
        return config.params or {}

    if config.method == "reasoning_param":
        # OpenAI style (o1, o3, GPT-5 series)
        effort = reasoning.get("effort", config.default_effort)
        return {"reasoning": {"effort": effort}}

    if config.method == "thinking_param":
        # Anthropic style (Claude 3.7, 4.x)
        params = (config.params or {}).copy()
        if reasoning.get("max_tokens"):
            params["budget_tokens"] = reasoning["max_tokens"]
        elif reasoning.get("effort") and config.effort_to_budget:
            params["budget_tokens"] = config.effort_to_budget.get(
                reasoning["effort"],
                config.effort_to_budget.get("medium", 8192),
            )
        return {"thinking": params}

    if config.method == "thinking_budget":
        # Alibaba/Qwen style
        if reasoning.get("max_tokens"):
            return {"thinking_budget": reasoning["max_tokens"]}
        elif reasoning.get("effort"):
            budget_map = {"low": 1024, "medium": 8192, "high": 32768}
            return {"thinking_budget": budget_map.get(reasoning["effort"], 8192)}
        return {}

    if config.method == "model_selection":
        # Kimi/Moonshot style - reasoning enabled by model name
        # No additional params needed
        return {}

    return {}


def extract_reasoning_content(
    response: dict[str, Any],
    provider_id: str,
) -> str | None:
    """
    Extract reasoning text from a provider response.

    Args:
        response: Raw API response dict.
        provider_id: Provider identifier.

    Returns:
        Reasoning text if available, None otherwise.
    """
    config = get_reasoning_config(provider_id)
    if not config or not config.response_field:
        return None

    # Standard OpenAI-compatible format
    choices = response.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})

    # Check for the reasoning field
    content = message.get(config.response_field)
    if content:
        return content

    # For Anthropic-style content blocks
    if config.method == "thinking_param":
        content_blocks = message.get("content", [])
        if isinstance(content_blocks, list):
            texts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    texts.append(block.get("thinking", ""))
            if texts:
                return "\n".join(texts)

    return None


def extract_streaming_reasoning_delta(
    chunk: dict[str, Any],
    provider_id: str,
) -> str:
    """
    Extract reasoning delta from a streaming chunk.

    Args:
        chunk: Streaming chunk dict.
        provider_id: Provider identifier.

    Returns:
        Reasoning delta string (empty if none).
    """
    config = get_reasoning_config(provider_id)
    if not config or not config.response_field:
        return ""

    choices = chunk.get("choices", [])
    if not choices:
        return ""

    delta = choices[0].get("delta", {})

    # Standard OpenAI-compatible streaming format
    content = delta.get(config.response_field, "")
    if content:
        return content

    # For Anthropic-style streaming
    if config.method == "thinking_param":
        # Anthropic sends thinking_delta events
        if delta.get("type") == "thinking_delta":
            return delta.get("thinking", "")

    return ""
