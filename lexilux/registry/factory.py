"""
Chat client factory with automatic model configuration.

Provides factory classes for creating pre-configured Chat instances
based on model specifications from the registry.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from lexilux.chat.client import Chat
from lexilux.chat.history import ChatHistory
from lexilux.chat.models import ChatResult, MessagesLike
from lexilux.chat.params import ChatParams
from lexilux.chat.streaming import StreamingIterator
from lexilux.registry.models import ModelSpec
from lexilux.registry.registry import ModelRegistry

if TYPE_CHECKING:
    from lexilux.chat.tools import Tool

logger = logging.getLogger(__name__)


class ConfiguredChat(Chat):
    """
    Chat client with model specification information.

    Extends the Chat class with:
    - Model specification metadata (capabilities, limits, modalities)
    - Convenience properties for checking model capabilities
    - Smart default parameter generation based on model specs

    This class is typically created via ChatFactory.create() rather than
    instantiated directly.

    Attributes:
        model_spec: The ModelSpec containing all model information.

    Examples:
        >>> # Created via factory
        >>> factory = ChatFactory()
        >>> chat = factory.create("gpt-4o", provider="openai", api_key="...")
        >>>
        >>> # Check capabilities
        >>> if chat.supports_tool_call:
        ...     result = chat("Call the weather function", tools=[...])
        >>>
        >>> # Get model limits
        >>> print(f"Context window: {chat.context_limit}")
        >>> print(f"Max output: {chat.output_limit}")
        >>>
        >>> # Use recommended defaults
        >>> params = chat.get_recommended_params()
        >>> result = chat("Hello!", params=params)
    """

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        **kwargs: Any,
    ):
        """
        Initialize ConfiguredChat.

        Args:
            model_spec: Model specification from the registry.
            **kwargs: Arguments passed to Chat.__init__.
        """
        super().__init__(**kwargs)
        self._model_spec = model_spec

    @property
    def model_spec(self) -> ModelSpec:
        """
        Get the model specification.

        Returns:
            ModelSpec containing all model information.
        """
        return self._model_spec

    @property
    def supports_tool_call(self) -> bool:
        """
        Whether the model supports function/tool calling.

        Returns:
            True if tool calling is supported.

        Examples:
            >>> if chat.supports_tool_call:
            ...     result = chat("Get weather", tools=[weather_tool])
        """
        return self._model_spec.capabilities.tool_call

    @property
    def supports_reasoning(self) -> bool:
        """
        Whether the model supports reasoning/chain-of-thought mode.

        Returns:
            True if reasoning mode is supported.
        """
        return self._model_spec.capabilities.reasoning

    @property
    def supports_structured_output(self) -> bool:
        """
        Whether the model supports structured output (JSON mode).

        Returns:
            True if structured output is supported.
        """
        return self._model_spec.capabilities.structured_output

    @property
    def supports_vision(self) -> bool:
        """
        Whether the model supports image input.

        Returns:
            True if vision/image input is supported.

        Examples:
            >>> if chat.supports_vision:
            ...     result = chat([{"type": "image_url", ...}])
        """
        return "image" in self._model_spec.modalities.input

    @property
    def supports_audio(self) -> bool:
        """
        Whether the model supports audio input.

        Returns:
            True if audio input is supported.
        """
        return "audio" in self._model_spec.modalities.input

    @property
    def supports_temperature(self) -> bool:
        """
        Whether the model supports temperature parameter.

        Returns:
            True if temperature is supported.
        """
        return self._model_spec.capabilities.temperature

    @property
    def context_limit(self) -> int:
        """
        Maximum context window size in tokens.

        Returns:
            Context window size.

        Examples:
            >>> print(f"Max context: {chat.context_limit} tokens")
        """
        return self._model_spec.limits.context

    @property
    def output_limit(self) -> int:
        """
        Maximum output length in tokens.

        Returns:
            Max output tokens.
        """
        return self._model_spec.limits.output

    @property
    def model_family(self) -> str | None:
        """
        Model family (e.g., "gpt", "claude", "qwen").

        Returns:
            Family name or None.
        """
        return self._model_spec.family

    @property
    def is_open_weights(self) -> bool:
        """
        Whether the model has open weights.

        Returns:
            True if open weights are available.
        """
        return self._model_spec.open_weights

    @property
    def input_modalities(self) -> tuple[str, ...]:
        """
        Supported input modalities.

        Returns:
            Tuple of modality strings (e.g., ("text", "image")).
        """
        return self._model_spec.modalities.input

    @property
    def output_modalities(self) -> tuple[str, ...]:
        """
        Supported output modalities.

        Returns:
            Tuple of modality strings.
        """
        return self._model_spec.modalities.output

    def get_recommended_params(
        self,
        *,
        max_tokens_ratio: float = 0.5,
        temperature: float = 0.7,
    ) -> ChatParams:
        """
        Get recommended default parameters based on model specs.

        Generates sensible default parameters based on the model's
        capabilities and limits.

        Args:
            max_tokens_ratio: Ratio of output_limit to use for max_tokens.
                Default is 0.5 (half of maximum output).
            temperature: Default temperature value.

        Returns:
            ChatParams with recommended defaults.

        Examples:
            >>> params = chat.get_recommended_params()
            >>> result = chat("Hello!", params=params)

            >>> # Custom ratio
            >>> params = chat.get_recommended_params(max_tokens_ratio=0.25)
        """
        # Calculate max_tokens based on model limit
        max_tokens = int(self._model_spec.limits.output * max_tokens_ratio)
        # Ensure reasonable minimum
        max_tokens = max(max_tokens, 256)
        # Cap at model's limit
        max_tokens = min(max_tokens, self._model_spec.limits.output)

        return ChatParams(
            temperature=temperature if self.supports_temperature else 0.0,
            max_tokens=max_tokens,
        )

    def __call__(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        user: str | None = None,
        n: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | Any | None = None,
        parallel_tool_calls: bool | None = None,
        params: ChatParams | None = None,
        extra: dict[str, Any] | None = None,
        return_raw: bool = False,
    ) -> ChatResult:
        """
        Make a chat completion request with smart defaults.

        If max_tokens is not specified and params is None, uses a
        reasonable default based on model's output limit.

        All arguments are the same as Chat.__call__.
        """
        # Apply smart defaults if no explicit params
        if params is None and max_tokens is None:
            # Use half of output limit as default, capped at 4096
            max_tokens = min(self._model_spec.limits.output // 2, 4096)
            max_tokens = max(max_tokens, 256)  # Minimum 256

        return super().__call__(
            messages,
            history=history,
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            n=n,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            params=params,
            extra=extra,
            return_raw=return_raw,
        )

    def stream(
        self,
        messages: MessagesLike,
        *,
        history: ChatHistory | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[int, float] | None = None,
        user: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | Any | None = None,
        parallel_tool_calls: bool | None = None,
        params: ChatParams | None = None,
        extra: dict[str, Any] | None = None,
        include_usage: bool = True,
        return_raw_events: bool = False,
    ) -> StreamingIterator:
        """
        Make a streaming chat completion request with smart defaults.

        If max_tokens is not specified and params is None, uses a
        reasonable default based on model's output limit.

        All arguments are the same as Chat.stream().
        """
        # Apply smart defaults if no explicit params
        if params is None and max_tokens is None:
            # Use half of output limit as default, capped at 4096
            max_tokens = min(self._model_spec.limits.output // 2, 4096)
            max_tokens = max(max_tokens, 256)  # Minimum 256

        return super().stream(
            messages,
            history=history,
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            params=params,
            extra=extra,
            include_usage=include_usage,
            return_raw_events=return_raw_events,
        )

    def __repr__(self) -> str:
        """String representation with model info."""
        return (
            f"ConfiguredChat("
            f"model={self.model!r}, "
            f"context={self.context_limit}, "
            f"output={self.output_limit}, "
            f"tool_call={self.supports_tool_call}, "
            f"reasoning={self.supports_reasoning}, "
            f"vision={self.supports_vision})"
        )


class ChatFactory:
    """
    Factory for creating pre-configured Chat clients.

    Creates Chat instances with automatic configuration based on
    model specifications from the registry. Handles API key lookup
    from environment variables and base URL resolution.

    Examples:
        >>> factory = ChatFactory()
        >>>
        >>> # Create with automatic configuration
        >>> chat = factory.create("gpt-4o", provider="openai", api_key="sk-...")
        >>>
        >>> # Check model capabilities
        >>> print(f"Context: {chat.context_limit}")
        >>> print(f"Supports tools: {chat.supports_tool_call}")
        >>>
        >>> # Use the chat
        >>> result = chat("Hello, world!")

        >>> # For unknown models (will use conservative defaults)
        >>> chat = factory.create(
        ...     "my-custom-model",
        ...     base_url="http://localhost:8080/v1",
        ...     suppress_unknown_warning=True,
        ... )

        >>> # Query available models
        >>> for model in factory.registry.models(provider="openai"):
        ...     print(model.name)
    """

    def __init__(self, registry: ModelRegistry | None = None):
        """
        Initialize the factory.

        Args:
            registry: Model registry instance. If None, uses the
                global singleton instance.

        Examples:
            >>> # Use default registry
            >>> factory = ChatFactory()

            >>> # Use custom registry
            >>> registry = ModelRegistry(data_path="/custom/models.json")
            >>> factory = ChatFactory(registry=registry)
        """
        self._registry = registry or ModelRegistry.get_instance()

    @property
    def registry(self) -> ModelRegistry:
        """
        Get the model registry.

        Returns:
            The ModelRegistry instance used by this factory.
        """
        return self._registry

    def create(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        suppress_unknown_warning: bool = False,
        # Chat parameters
        timeout_s: float = 60.0,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        max_retries: int = 0,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ) -> ConfiguredChat:
        """
        Create a pre-configured Chat instance.

        Automatically resolves:
        - Model specification from registry
        - Base URL from provider configuration
        - API key from environment variables

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus").
            provider: Provider identifier. If None, auto-detected from model.
            base_url: API base URL. If None, resolved from provider config.
            api_key: API key. If None, read from environment variables.
            suppress_unknown_warning: Don't warn for unknown models.
            timeout_s: Request timeout in seconds.
            connect_timeout_s: Connection timeout (overrides timeout_s).
            read_timeout_s: Read timeout (overrides timeout_s).
            max_retries: Maximum retry attempts for failed requests.
            pool_connections: Number of connection pools.
            pool_maxsize: Maximum connections per pool.
            headers: Additional request headers.
            proxies: Proxy configuration.

        Returns:
            ConfiguredChat instance with model specification.

        Raises:
            ValueError: If base_url cannot be resolved.

        Examples:
            >>> factory = ChatFactory()
            >>>
            >>> # Basic usage (auto-resolve everything)
            >>> chat = factory.create("gpt-4o", provider="openai")
            >>>
            >>> # With explicit configuration
            >>> chat = factory.create(
            ...     "gpt-4o",
            ...     provider="openai",
            ...     api_key="sk-...",
            ...     timeout_s=120.0,
            ... )
            >>>
            >>> # Custom/unknown model
            >>> chat = factory.create(
            ...     "my-model",
            ...     base_url="http://localhost:8080/v1",
            ...     suppress_unknown_warning=True,
            ... )
        """
        # Get model specification
        spec = self._registry.get(
            model,
            provider=provider,
            suppress_unknown_warning=suppress_unknown_warning,
        )

        # Resolve provider_id
        resolved_provider = provider or spec.provider_id

        # Resolve base_url
        if base_url is None:
            provider_spec = self._registry.get_provider(resolved_provider)
            if provider_spec and provider_spec.api_base:
                base_url = provider_spec.api_base
            else:
                raise ValueError(
                    f"base_url is required for model '{model}' "
                    f"(provider '{resolved_provider}' not found or has no API base URL). "
                    f"Please provide base_url explicitly."
                )

        # Resolve api_key from environment
        if api_key is None:
            provider_spec = self._registry.get_provider(resolved_provider)
            if provider_spec:
                for env_var in provider_spec.env_vars:
                    api_key = os.environ.get(env_var)
                    if api_key:
                        logger.debug(f"Using API key from {env_var}")
                        break

        # Create ConfiguredChat instance
        return ConfiguredChat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            model_spec=spec,
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            max_retries=max_retries,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            headers=headers,
            proxies=proxies,
        )

    def list_providers(self) -> list[str]:
        """
        List all available provider IDs.

        Returns:
            List of provider identifiers.

        Examples:
            >>> print(factory.list_providers())
            ['openai', 'anthropic', 'google', ...]
        """
        return self._registry.provider_ids()

    def list_models(self, provider: str | None = None) -> list[str]:
        """
        List available model IDs.

        Args:
            provider: If specified, only list models from this provider.

        Returns:
            List of model identifiers.

        Examples:
            >>> print(factory.list_models(provider="openai"))
            ['gpt-4o', 'gpt-4-turbo', ...]
        """
        return self._registry.model_ids(provider=provider)

    def get_model_spec(
        self,
        model: str,
        provider: str | None = None,
    ) -> ModelSpec:
        """
        Get model specification without creating a client.

        Args:
            model: Model identifier.
            provider: Provider identifier (optional).

        Returns:
            ModelSpec for the model.

        Examples:
            >>> spec = factory.get_model_spec("gpt-4o")
            >>> print(f"Context: {spec.limits.context}")
        """
        return self._registry.get(
            model, provider=provider, suppress_unknown_warning=True
        )

    def search_models(
        self,
        *,
        supports_tool_call: bool | None = None,
        supports_reasoning: bool | None = None,
        supports_vision: bool | None = None,
        min_context: int | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Iterator[ModelSpec]:
        """
        Search for models matching criteria.

        Args:
            supports_tool_call: Filter by tool calling support.
            supports_reasoning: Filter by reasoning support.
            supports_vision: Filter by vision support.
            min_context: Minimum context window.
            provider: Limit to specific provider.
            **kwargs: Additional search criteria.

        Yields:
            Matching ModelSpec objects.

        Examples:
            >>> # Find reasoning models
            >>> for model in factory.search_models(supports_reasoning=True):
            ...     print(model.name)

            >>> # Find large context models with tools
            >>> for model in factory.search_models(min_context=100000, supports_tool_call=True):
            ...     print(f"{model.name}: {model.limits.context}")
        """
        return self._registry.search(
            supports_tool_call=supports_tool_call,
            supports_reasoning=supports_reasoning,
            supports_vision=supports_vision,
            min_context=min_context,
            provider=provider,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ChatFactory(registry={self._registry!r})"
