"""
Provider configuration registry.

Maps provider IDs to their reasoning configurations and provides
URL-based provider detection.
"""

from lexilux.providers.base import ReasoningConfig

# Provider ID to ReasoningConfig mapping
REASONING_CONFIGS: dict[str, ReasoningConfig] = {
    # DeepSeek: Enable via extra_body, reasoning visible in reasoning_content
    "deepseek": ReasoningConfig(
        method="extra_body",
        response_field="reasoning_content",
        params={"thinking": {"type": "enabled"}},
    ),
    # OpenAI: Enable via reasoning parameter, reasoning hidden
    "openai": ReasoningConfig(
        method="reasoning_param",
        response_field=None,  # Hidden by default
        default_effort="medium",
    ),
    # Anthropic: Enable via thinking parameter with budget_tokens
    "anthropic": ReasoningConfig(
        method="thinking_param",
        response_field="thinking",
        supports_budget=True,
        effort_to_budget={
            "low": 1024,
            "medium": 8192,
            "high": 32768,
            "xhigh": 65536,
        },
        params={"type": "enabled"},
    ),
    # Moonshot/Kimi: Reasoning enabled by model selection
    "moonshotai": ReasoningConfig(
        method="model_selection",
        response_field="reasoning_content",
    ),
    # Zhipu/GLM: Enable via extra_body
    "zhipuai": ReasoningConfig(
        method="extra_body",
        response_field="reasoning_content",
        params={"thinking": {"type": "enabled"}},
    ),
    # Minimax: OpenAI-compatible with reasoning
    "minimax": ReasoningConfig(
        method="extra_body",
        response_field="reasoning_content",
        params={"thinking": {"type": "enabled"}},
    ),
    # Alibaba/Qwen: Thinking budget support
    "alibaba": ReasoningConfig(
        method="thinking_budget",
        response_field="reasoning_content",
        supports_budget=True,
    ),
}

# URL patterns for provider detection
PROVIDER_URL_PATTERNS: dict[str, list[str]] = {
    "deepseek": ["deepseek.com"],
    "openai": ["openai.com"],
    "anthropic": ["anthropic.com", "claude.ai"],
    "moonshotai": ["moonshot.cn", "moonshot.ai"],
    "zhipuai": ["bigmodel.cn", "zhipuai.cn", "zhipu.ai"],
    "minimax": ["minimax.chat"],
    "alibaba": ["dashscope.aliyuncs.com"],
}


def get_reasoning_config(provider_id: str) -> ReasoningConfig | None:
    """
    Get reasoning configuration for a provider.

    Args:
        provider_id: Provider identifier (e.g., "openai", "deepseek").

    Returns:
        ReasoningConfig if provider has reasoning support, None otherwise.
    """
    return REASONING_CONFIGS.get(provider_id)


def detect_provider_from_url(base_url: str) -> str | None:
    """
    Detect provider ID from base URL.

    Args:
        base_url: API base URL.

    Returns:
        Provider ID if detected, None otherwise.
    """
    base_url_lower = base_url.lower()
    for provider_id, patterns in PROVIDER_URL_PATTERNS.items():
        for pattern in patterns:
            if pattern in base_url_lower:
                return provider_id
    return None
