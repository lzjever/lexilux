"""
Tokenizer API client (optional dependency on transformers).

Provides local tokenization with support for offline/online modes and automatic model downloading.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, TYPE_CHECKING

from lexilux.usage import Usage, ResultBase, Json

if TYPE_CHECKING:
    pass

# Type alias
TokenizerMode = Literal["online", "auto_offline", "force_offline"]


class TokenizeResult(ResultBase):
    """
    Tokenize result.

    Attributes:
        input_ids: List of token IDs (List[List[int]] for batch input).
        attention_mask: Attention mask (List[List[int]] for batch input, optional).
        usage: Usage statistics (at least input_tokens is provided).
        raw: Raw tokenizer output.

    Examples:
        >>> result = tokenizer("Hello, world!")
        >>> print(result.input_ids)  # [[15496, 11, 1917, 0]]
        >>> print(result.usage.input_tokens)  # 4
    """

    def __init__(
        self,
        *,
        input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]],
        usage: Usage,
        raw: Optional[Json] = None,
    ):
        """
        Initialize TokenizeResult.

        Args:
            input_ids: List of token ID sequences.
            attention_mask: Attention mask sequences (optional).
            usage: Usage statistics.
            raw: Raw tokenizer output.
        """
        super().__init__(usage=usage, raw=raw)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TokenizeResult(input_ids=[{len(self.input_ids)} sequences], usage={self.usage!r})"


class Tokenizer:
    """
    Tokenizer client (uses transformers library).

    Provides local tokenization with support for:
    - Online mode: Always allows network access for downloading models
    - Auto-offline mode: Uses local cache if available, downloads if not
    - Force-offline mode: Only uses local cache, fails if model not found

    Examples:
        >>> # Auto-offline mode (recommended)
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", mode="auto_offline")
        >>> result = tokenizer("Hello, world!")
        >>> print(result.usage.input_tokens)

        >>> # Force-offline mode (for air-gapped environments)
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", mode="force_offline", cache_dir="/models/hf")
        >>> result = tokenizer("Hello, world!")

        >>> # Online mode
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", mode="online")
        >>> result = tokenizer("Hello, world!")
    """

    def __init__(
        self,
        model: str,
        *,
        cache_dir: Optional[str] = None,
        mode: TokenizerMode = "auto_offline",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        require_transformers: bool = True,
    ):
        """
        Initialize Tokenizer client.

        Args:
            model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct").
            cache_dir: Directory to cache models (defaults to HuggingFace cache).
            mode: Tokenizer mode ("online", "auto_offline", "force_offline").
            revision: Model revision/branch/tag (optional).
            trust_remote_code: Whether to allow remote code execution.
            require_transformers: If True, raise error immediately if transformers not installed.
                                 If False, delay error until first use.

        Raises:
            ImportError: If transformers is not installed and require_transformers=True.
        """
        self.model = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.require_transformers = require_transformers

        # Lazy import transformers
        self._tokenizer = None
        self._transformers_available = False

        # Check transformers availability
        try:
            import transformers  # noqa: F401

            self._transformers_available = True
        except ImportError:
            if require_transformers:
                raise ImportError(
                    "transformers library is required for Tokenizer. "
                    "Install it with: pip install lexilux[tokenizer] (or lexilux[token]) or pip install transformers"
                )
            # If require_transformers=False, we'll check again on first use

    def _ensure_tokenizer(self):
        """
        Ensure tokenizer is loaded (lazy loading).

        Raises:
            ImportError: If transformers is not available.
            OSError: If model cannot be loaded (e.g., force_offline mode and model not found).
        """
        if self._tokenizer is not None:
            return

        # Check transformers availability
        if not self._transformers_available:
            try:
                import transformers  # noqa: F401

                self._transformers_available = True
            except ImportError:
                raise ImportError(
                    "transformers library is required for Tokenizer. "
                    "Install it with: pip install lexilux[tokenizer] or pip install transformers"
                )

        # Import transformers components
        from transformers import AutoTokenizer

        # Determine local_files_only based on mode
        if self.mode == "force_offline":
            local_files_only = True
        elif self.mode == "auto_offline":
            # Try local first, fallback to online
            local_files_only = None  # Will try local first
        else:  # online
            local_files_only = False

        # Load tokenizer
        try:
            if self.mode == "auto_offline":
                # Try local first
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model,
                        cache_dir=self.cache_dir,
                        revision=self.revision,
                        trust_remote_code=self.trust_remote_code,
                        local_files_only=True,
                    )
                except (OSError, ValueError):
                    # Local not available, try online
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model,
                        cache_dir=self.cache_dir,
                        revision=self.revision,
                        trust_remote_code=self.trust_remote_code,
                        local_files_only=False,
                    )
            else:
                # force_offline or online
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model,
                    cache_dir=self.cache_dir,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=local_files_only,
                )
        except Exception as e:
            if self.mode == "force_offline":
                raise OSError(
                    f"Model '{self.model}' not found in local cache. "
                    f"Force-offline mode requires the model to be pre-downloaded. "
                    f"Cache dir: {self.cache_dir or 'default HuggingFace cache'}"
                ) from e
            raise

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        *,
        add_special_tokens: bool = True,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        return_attention_mask: bool = True,
        extra: Optional[Json] = None,
        return_raw: bool = False,
    ) -> TokenizeResult:
        """
        Tokenize text.

        Args:
            text: Single text string or sequence of text strings.
            add_special_tokens: Whether to add special tokens (e.g., BOS, EOS).
            truncation: Truncation strategy (True, False, or "longest_first", etc.).
            max_length: Maximum sequence length.
            padding: Padding strategy (True, False, or "max_length", etc.).
            return_attention_mask: Whether to return attention mask.
            extra: Additional tokenizer parameters.
            return_raw: Whether to include raw tokenizer output.

        Returns:
            TokenizeResult with input_ids, attention_mask, and usage.

        Raises:
            ImportError: If transformers is not available.
            OSError: If model cannot be loaded (force_offline mode).
        """
        # Ensure tokenizer is loaded
        self._ensure_tokenizer()

        # Normalize input to list
        is_single = isinstance(text, str)
        text_list = [text] if is_single else list(text)

        if not text_list:
            raise ValueError("Text cannot be empty")

        # Prepare tokenizer arguments
        tokenizer_kwargs: Dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "return_attention_mask": return_attention_mask,
        }

        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length

        if extra:
            tokenizer_kwargs.update(extra)

        # Tokenize
        encoded = self._tokenizer(text_list, **tokenizer_kwargs)

        # Extract results
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask") if return_attention_mask else None

        # Calculate usage (total tokens across all sequences)
        total_tokens = sum(len(ids) for ids in input_ids)

        # Create usage
        usage = Usage(
            input_tokens=total_tokens,
            output_tokens=None,  # Not applicable for tokenization
            total_tokens=total_tokens,
        )

        # Return result
        return TokenizeResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            usage=usage,
            raw=encoded if return_raw else {},
        )
