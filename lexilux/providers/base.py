"""
Base types for provider configurations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningConfig:
    """
    Configuration for how a provider handles reasoning/extended thinking.

    Attributes:
        method: How to enable reasoning in the request.
            - "extra_body": Add params to extra_body (DeepSeek style)
            - "reasoning_param": Use reasoning parameter (OpenAI style)
            - "thinking_param": Use thinking parameter (Anthropic style)
            - "model_selection": Reasoning enabled by model name (Kimi style)
        response_field: Field name in response containing reasoning content.
            None if reasoning is hidden (OpenAI o-series).
        default_effort: Default effort level for providers that support it.
        supports_budget: Whether the provider supports budget_tokens.
        effort_to_budget: Mapping from effort levels to token budgets.
        params: Provider-specific parameters to inject when enabling reasoning.
    """

    method: str
    response_field: str | None = None
    default_effort: str = "medium"
    supports_budget: bool = False
    effort_to_budget: dict | None = None
    params: dict | None = None
