"""
Model Registry for AI model specifications.

Provides centralized access to model information, capabilities,
and provider details from the models.dev database.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from typing import TYPE_CHECKING, Iterator

from lexilux.registry.models import (
    ModelCapabilities,
    ModelCost,
    ModelLimits,
    ModelModalities,
    ModelSpec,
    ProviderSpec,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Conservative default configuration for unknown models
_CONSERVATIVE_CAPABILITIES = ModelCapabilities(
    tool_call=False,
    reasoning=False,
    structured_output=False,
    attachment=False,
    temperature=True,
    interleaved=False,
)

_CONSERVATIVE_LIMITS = ModelLimits(
    context=8192,
    output=4096,
)

_CONSERVATIVE_MODALITIES = ModelModalities(
    input=("text",),
    output=("text",),
)


class ModelRegistry:
    """
    AI Model Registry.

    Provides model information lookup, provider queries, and model search
    functionality based on the models.dev database.

    The registry loads model data from a bundled JSON file and provides
    efficient access to model specifications and provider information.

    Examples:
        >>> registry = ModelRegistry()
        >>>
        >>> # Query all providers
        >>> for provider in registry.providers():
        ...     print(provider.name)
        OpenAI
        Anthropic
        ...

        >>> # Query models from a specific provider
        >>> for model in registry.models(provider="openai"):
        ...     print(f"{model.name}: {model.limits.context}k context")

        >>> # Get a specific model's specification
        >>> spec = registry.get("gpt-4o", provider="openai")
        >>> print(spec.capabilities.tool_call)  # True

        >>> # Search for models with specific capabilities
        >>> for model in registry.search(supports_reasoning=True):
        ...     print(f"{model.name} ({model.provider_id})")

    Thread Safety:
        The registry is designed to be thread-safe for read operations.
        The singleton instance can be safely accessed from multiple threads.
    """

    _instance: ModelRegistry | None = None

    def __init__(self, data_path: str | None = None):
        """
        Initialize the model registry.

        Args:
            data_path: Custom path to the models JSON file.
                If None, uses the bundled data file.

        Examples:
            >>> # Use bundled data
            >>> registry = ModelRegistry()

            >>> # Use custom data file
            >>> registry = ModelRegistry(data_path="/path/to/models.json")
        """
        self._providers: dict[str, ProviderSpec] = {}
        self._models: dict[
            str, dict[str, ModelSpec]
        ] = {}  # provider_id -> model_id -> spec
        self._model_index: dict[str, list[str]] = {}  # model_id -> list of provider_ids
        self._loaded = False
        self._load_data(data_path)

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """
        Get the singleton registry instance.

        Returns a shared registry instance, creating it if necessary.
        This is the recommended way to access the registry in most cases.

        Returns:
            The singleton ModelRegistry instance.

        Examples:
            >>> registry = ModelRegistry.get_instance()
            >>> spec = registry.get("gpt-4o")
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when you need to reload data.
        """
        cls._instance = None

    def _load_data(self, data_path: str | None = None) -> None:
        """Load model data from file."""
        data: dict = {}

        if data_path:
            try:
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load models data from {data_path}: {e}")
                return
        else:
            # Load from package data
            try:
                data_file = resources.files("lexilux.data").joinpath("models.json")
                data = json.loads(data_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load bundled models data: {e}")
                return

        self._parse_data(data)
        self._loaded = True

    def _parse_data(self, data: dict) -> None:
        """Parse models.dev API data format."""
        for provider_id, provider_data in data.items():
            if not isinstance(provider_data, dict):
                continue

            # Parse provider
            provider = ProviderSpec(
                id=provider_id,
                name=provider_data.get("name", provider_id),
                api_base=provider_data.get("api", ""),
                doc_url=provider_data.get("doc"),
                env_vars=tuple(provider_data.get("env", [])),
                npm_package=provider_data.get("npm"),
            )
            self._providers[provider_id] = provider

            # Parse models
            self._models[provider_id] = {}
            models_data = provider_data.get("models", {})

            if not isinstance(models_data, dict):
                continue

            for model_id, model_data in models_data.items():
                if not isinstance(model_data, dict):
                    continue

                spec = self._parse_model(model_data, provider_id)
                self._models[provider_id][model_id] = spec

                # Build model index for fast lookup
                if model_id not in self._model_index:
                    self._model_index[model_id] = []
                self._model_index[model_id].append(provider_id)

    def _parse_model(self, data: dict, provider_id: str) -> ModelSpec:
        """Parse a single model's data."""
        # Parse capabilities
        interleaved = data.get("interleaved", False)
        capabilities = ModelCapabilities(
            tool_call=bool(data.get("tool_call", False)),
            reasoning=bool(data.get("reasoning", False)),
            structured_output=bool(data.get("structured_output", False)),
            attachment=bool(data.get("attachment", False)),
            temperature=bool(data.get("temperature", True)),
            interleaved=interleaved if isinstance(interleaved, (bool, dict)) else False,
        )

        # Parse limits
        limit_data = data.get("limit", {})
        if not isinstance(limit_data, dict):
            limit_data = {}
        limits = ModelLimits(
            context=int(limit_data.get("context", 8192) or 8192),
            output=int(limit_data.get("output", 4096) or 4096),
        )

        # Parse modalities
        modality_data = data.get("modalities", {})
        if not isinstance(modality_data, dict):
            modality_data = {}
        modalities = ModelModalities(
            input=tuple(modality_data.get("input", ["text"])),
            output=tuple(modality_data.get("output", ["text"])),
        )

        # Parse cost
        cost_data = data.get("cost", {})
        if not isinstance(cost_data, dict):
            cost_data = {}
        cost = ModelCost(
            input=float(cost_data.get("input", 0.0) or 0.0),
            output=float(cost_data.get("output", 0.0) or 0.0),
        )

        # Parse status
        status_value = data.get("status")
        status: str = "deprecated" if status_value == "deprecated" else "active"

        return ModelSpec(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            provider_id=provider_id,
            capabilities=capabilities,
            limits=limits,
            modalities=modalities,
            cost=cost,
            family=data.get("family"),
            knowledge_cutoff=data.get("knowledge"),
            release_date=data.get("release_date"),
            open_weights=bool(data.get("open_weights", False)),
            status=status,  # type: ignore[arg-type]
        )

    # === Query API ===

    @property
    def is_loaded(self) -> bool:
        """Whether the registry has successfully loaded data."""
        return self._loaded

    def providers(self) -> Iterator[ProviderSpec]:
        """
        Iterate over all providers.

        Yields:
            ProviderSpec objects for each registered provider.

        Examples:
            >>> for provider in registry.providers():
            ...     print(f"{provider.name}: {provider.api_base}")
        """
        yield from self._providers.values()

    def get_provider(self, provider_id: str) -> ProviderSpec | None:
        """
        Get a specific provider by ID.

        Args:
            provider_id: Provider identifier (e.g., "openai", "anthropic").

        Returns:
            ProviderSpec if found, None otherwise.

        Examples:
            >>> provider = registry.get_provider("openai")
            >>> if provider:
            ...     print(provider.api_base)
        """
        return self._providers.get(provider_id)

    def provider_ids(self) -> list[str]:
        """
        Get list of all provider IDs.

        Returns:
            List of provider identifiers.

        Examples:
            >>> print(registry.provider_ids())
            ['openai', 'anthropic', 'google', ...]
        """
        return list(self._providers.keys())

    def models(self, provider: str | None = None) -> Iterator[ModelSpec]:
        """
        Iterate over models.

        Args:
            provider: If specified, only return models from this provider.

        Yields:
            ModelSpec objects.

        Examples:
            >>> # All models
            >>> for model in registry.models():
            ...     print(model.name)

            >>> # Models from specific provider
            >>> for model in registry.models(provider="openai"):
            ...     print(model.name)
        """
        if provider:
            if provider in self._models:
                yield from self._models[provider].values()
        else:
            for provider_models in self._models.values():
                yield from provider_models.values()

    def model_ids(self, provider: str | None = None) -> list[str]:
        """
        Get list of model IDs.

        Args:
            provider: If specified, only return model IDs from this provider.

        Returns:
            List of model identifiers.

        Examples:
            >>> print(registry.model_ids(provider="openai"))
            ['gpt-4o', 'gpt-4-turbo', ...]
        """
        if provider:
            if provider in self._models:
                return list(self._models[provider].keys())
            return []
        return list(self._model_index.keys())

    def get(
        self,
        model_id: str,
        provider: str | None = None,
        *,
        suppress_unknown_warning: bool = False,
    ) -> ModelSpec:
        """
        Get a model's specification.

        If the model is not found, returns a conservative default configuration
        and logs a warning (unless suppressed).

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "claude-3-opus").
            provider: Provider identifier. If None, searches all providers.
            suppress_unknown_warning: If True, don't log warning for unknown models.

        Returns:
            ModelSpec for the model. If not found, returns conservative defaults.

        Examples:
            >>> # Get known model
            >>> spec = registry.get("gpt-4o", provider="openai")
            >>> print(spec.limits.context)  # 128000

            >>> # Unknown model with warning
            >>> spec = registry.get("my-custom-model")
            >>> # Warning: Model 'my-custom-model' not found...

            >>> # Unknown model without warning
            >>> spec = registry.get("my-custom-model", suppress_unknown_warning=True)
        """
        # Search with specified provider
        if provider:
            if provider in self._models and model_id in self._models[provider]:
                return self._models[provider][model_id]
        else:
            # Search in model index
            if model_id in self._model_index:
                # Return first match
                first_provider = self._model_index[model_id][0]
                return self._models[first_provider][model_id]

        # Not found - return conservative defaults
        if not suppress_unknown_warning:
            logger.warning(
                f"Model '{model_id}' not found in registry. "
                f"Using conservative defaults (8K context, text-only, no tool_call). "
                f"Use suppress_unknown_warning=True to silence this warning."
            )

        return ModelSpec(
            id=model_id,
            name=model_id,
            provider_id=provider or "unknown",
            capabilities=_CONSERVATIVE_CAPABILITIES,
            limits=_CONSERVATIVE_LIMITS,
            modalities=_CONSERVATIVE_MODALITIES,
        )

    def get_providers_for_model(self, model_id: str) -> list[str]:
        """
        Get list of providers that offer a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            List of provider IDs that offer this model.

        Examples:
            >>> providers = registry.get_providers_for_model("llama-3.1-70b")
            >>> print(providers)
            ['groq', 'fireworks-ai', 'togetherai', ...]
        """
        return self._model_index.get(model_id, [])

    def search(
        self,
        *,
        family: str | None = None,
        supports_tool_call: bool | None = None,
        supports_reasoning: bool | None = None,
        supports_vision: bool | None = None,
        supports_audio: bool | None = None,
        min_context: int | None = None,
        max_context: int | None = None,
        provider: str | None = None,
        open_weights: bool | None = None,
        status: str | None = None,
    ) -> Iterator[ModelSpec]:
        """
        Search for models matching specific criteria.

        All criteria are optional. Models must match ALL specified criteria.

        Args:
            family: Model family (e.g., "gpt", "claude", "qwen", "llama").
            supports_tool_call: Filter by tool/function calling support.
            supports_reasoning: Filter by reasoning mode support.
            supports_vision: Filter by image input support.
            supports_audio: Filter by audio input support.
            min_context: Minimum context window size.
            max_context: Maximum context window size.
            provider: Limit search to specific provider.
            open_weights: Filter by open weights availability.
            status: Filter by status ("active" or "deprecated").

        Yields:
            ModelSpec objects matching all criteria.

        Examples:
            >>> # Find reasoning models
            >>> for model in registry.search(supports_reasoning=True):
            ...     print(f"{model.name} ({model.provider_id})")

            >>> # Find large context models with tool calling
            >>> for model in registry.search(min_context=100000, supports_tool_call=True):
            ...     print(f"{model.name}: {model.limits.context}k")

            >>> # Find open-weight vision models
            >>> for model in registry.search(supports_vision=True, open_weights=True):
            ...     print(model.name)
        """
        for model in self.models(provider=provider):
            # Family filter
            if family is not None and model.family != family:
                continue

            # Capability filters
            if supports_tool_call is not None:
                if model.capabilities.tool_call != supports_tool_call:
                    continue

            if supports_reasoning is not None:
                if model.capabilities.reasoning != supports_reasoning:
                    continue

            if supports_vision is not None:
                has_vision = "image" in model.modalities.input
                if has_vision != supports_vision:
                    continue

            if supports_audio is not None:
                has_audio = "audio" in model.modalities.input
                if has_audio != supports_audio:
                    continue

            # Context filters
            if min_context is not None and model.limits.context < min_context:
                continue

            if max_context is not None and model.limits.context > max_context:
                continue

            # Open weights filter
            if open_weights is not None and model.open_weights != open_weights:
                continue

            # Status filter
            if status is not None and model.status != status:
                continue

            yield model

    def count(self, provider: str | None = None) -> int:
        """
        Count total number of models.

        Args:
            provider: If specified, count only models from this provider.

        Returns:
            Number of models.

        Examples:
            >>> print(f"Total models: {registry.count()}")
            >>> print(f"OpenAI models: {registry.count(provider='openai')}")
        """
        if provider:
            return len(self._models.get(provider, {}))
        return sum(len(models) for models in self._models.values())

    def provider_count(self) -> int:
        """
        Count total number of providers.

        Returns:
            Number of providers.
        """
        return len(self._providers)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModelRegistry(providers={self.provider_count()}, models={self.count()})"
        )
