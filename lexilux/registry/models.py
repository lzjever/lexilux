"""
Model and Provider specification data classes.

Defines immutable data structures for AI model specifications,
provider information, and model capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelCost:
    """
    Model pricing information (per million tokens).

    Attributes:
        input: Cost per million input tokens (USD).
        output: Cost per million output tokens (USD).

    Examples:
        >>> cost = ModelCost(input=2.5, output=10.0)
        >>> print(f"Input: ${cost.input}/M, Output: ${cost.output}/M")
    """

    input: float = 0.0
    output: float = 0.0


@dataclass(frozen=True)
class ModelLimits:
    """
    Model token limits.

    Attributes:
        context: Maximum context window size in tokens.
        output: Maximum output length in tokens.

    Examples:
        >>> limits = ModelLimits(context=128000, output=8192)
        >>> print(f"Context: {limits.context}, Max output: {limits.output}")
    """

    context: int = 8192
    output: int = 4096


@dataclass(frozen=True)
class ModelModalities:
    """
    Supported input/output modalities.

    Attributes:
        input: Tuple of supported input types ("text", "image", "audio").
        output: Tuple of supported output types ("text", "json", etc.).

    Examples:
        >>> modalities = ModelModalities(input=("text", "image"), output=("text",))
        >>> "image" in modalities.input  # True - supports vision
    """

    input: tuple[str, ...] = ("text",)
    output: tuple[str, ...] = ("text",)


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Model capabilities and feature support.

    Attributes:
        tool_call: Whether the model supports function/tool calling.
        reasoning: Whether the model supports reasoning/chain-of-thought mode.
        structured_output: Whether the model supports structured output (JSON mode).
        attachment: Whether the model supports file attachments.
        temperature: Whether the model supports temperature parameter.
        interleaved: Interleaved content support (bool or dict with config).

    Examples:
        >>> caps = ModelCapabilities(tool_call=True, reasoning=True)
        >>> if caps.tool_call:
        ...     print("Model supports function calling")
    """

    tool_call: bool = False
    reasoning: bool = False
    structured_output: bool = False
    attachment: bool = False
    temperature: bool = True
    interleaved: bool | dict[str, Any] = False


@dataclass(frozen=True)
class ModelSpec:
    """
    Complete model specification.

    Contains all information about an AI model including its capabilities,
    limits, supported modalities, pricing, and metadata.

    Attributes:
        id: Unique model identifier (e.g., "gpt-4o", "claude-3-opus").
        name: Human-readable model name.
        provider_id: Provider identifier (e.g., "openai", "anthropic").
        capabilities: Model capabilities (tool_call, reasoning, etc.).
        limits: Token limits (context window, max output).
        modalities: Supported input/output types.
        cost: Pricing information.
        family: Model family (e.g., "gpt", "claude", "qwen").
        knowledge_cutoff: Knowledge cutoff date (YYYY-MM format).
        release_date: Model release date.
        open_weights: Whether the model has open weights.
        status: Model status ("active" or "deprecated").

    Examples:
        >>> spec = ModelSpec(
        ...     id="gpt-4o",
        ...     name="GPT-4o",
        ...     provider_id="openai",
        ...     capabilities=ModelCapabilities(tool_call=True),
        ...     limits=ModelLimits(context=128000, output=16384),
        ... )
        >>> print(f"{spec.name}: {spec.limits.context}k context")
    """

    id: str
    name: str
    provider_id: str

    # Capabilities
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    # Limits
    limits: ModelLimits = field(default_factory=ModelLimits)

    # Modalities
    modalities: ModelModalities = field(default_factory=ModelModalities)

    # Cost
    cost: ModelCost = field(default_factory=ModelCost)

    # Metadata
    family: str | None = None
    knowledge_cutoff: str | None = None
    release_date: str | None = None
    open_weights: bool = False
    status: Literal["active", "deprecated"] = "active"

    @property
    def supports_tool_call(self) -> bool:
        """Whether the model supports function/tool calling."""
        return self.capabilities.tool_call

    @property
    def supports_reasoning(self) -> bool:
        """Whether the model supports reasoning mode."""
        return self.capabilities.reasoning

    @property
    def supports_vision(self) -> bool:
        """Whether the model supports image input."""
        return "image" in self.modalities.input

    @property
    def supports_audio(self) -> bool:
        """Whether the model supports audio input."""
        return "audio" in self.modalities.input

    @property
    def context_window(self) -> int:
        """Context window size in tokens."""
        return self.limits.context

    @property
    def max_output(self) -> int:
        """Maximum output length in tokens."""
        return self.limits.output


@dataclass(frozen=True)
class ProviderSpec:
    """
    Provider specification.

    Contains information about an AI API provider including
    connection details and required configuration.

    Attributes:
        id: Unique provider identifier (e.g., "openai", "anthropic").
        name: Human-readable provider name.
        api_base: Base URL for the API endpoint.
        doc_url: URL to provider documentation.
        env_vars: Tuple of environment variable names for authentication.
        npm_package: NPM package name for AI SDK integration.

    Examples:
        >>> provider = ProviderSpec(
        ...     id="openai",
        ...     name="OpenAI",
        ...     api_base="https://api.openai.com/v1",
        ...     env_vars=("OPENAI_API_KEY",),
        ... )
        >>> print(f"Use {provider.env_vars[0]} for authentication")
    """

    id: str
    name: str
    api_base: str
    doc_url: str | None = None
    env_vars: tuple[str, ...] = ()
    npm_package: str | None = None
