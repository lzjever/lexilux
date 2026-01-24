"""
Tokenizer API client (optional dependency on transformers).

Provides local tokenization with support for offline/online modes and automatic model downloading.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lexilux.usage import Json, ResultBase, Usage

if TYPE_CHECKING:
    pass


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
        input_ids: list[list[int]],
        attention_mask: list[list[int]] | None,
        usage: Usage,
        raw: Json | None = None,
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
    - Offline mode (offline=True): Only uses local cache, fails if model not found
    - Online mode (offline=False): Prioritizes local cache, downloads if not found

    Examples:
        >>> # Offline mode (for air-gapped environments)
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", offline=True, cache_dir="/models/hf")
        >>> result = tokenizer("Hello, world!")
        >>> print(result.usage.input_tokens)

        >>> # Online mode (default, downloads if needed)
        >>> tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", offline=False)
        >>> result = tokenizer("Hello, world!")
    """

    def __init__(
        self,
        model: str,
        *,
        cache_dir: str | None = None,
        offline: bool = False,
        revision: str | None = None,
        trust_remote_code: bool = False,
        require_transformers: bool = True,
    ):
        """
        Initialize Tokenizer client.

        Args:
            model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct").
            cache_dir: Directory to cache models (defaults to HuggingFace cache).
                      Supports "~" for home directory expansion.
            offline: If True, only use local cache (fail if not found).
                     If False, prioritize local cache, download if not found.
            revision: Model revision/branch/tag (optional).
            trust_remote_code: Whether to allow remote code execution.
            require_transformers: If True, raise error immediately if transformers not installed.
                                 If False, delay error until first use.

        Raises:
            ImportError: If transformers is not installed and require_transformers=True.
        """
        self.model = model
        # Expand "~" to home directory if cache_dir is provided
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self.offline = offline
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

    @staticmethod
    def list_tokenizer_files(
        model: str,
        *,
        revision: str | None = None,
    ) -> list[str]:
        """
        List tokenizer-related files for a given model.

        This method queries the HuggingFace Hub to identify which files
        are needed for tokenization, without downloading them.

        Args:
            model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct").
            revision: Model revision/branch/tag (optional).

        Returns:
            List of file paths that are tokenizer-related.

        Raises:
            ImportError: If huggingface_hub is not installed.
            Exception: If unable to list files from HuggingFace Hub.

        Example:
            >>> files = Tokenizer.list_tokenizer_files("Qwen/Qwen2.5-7B-Instruct")
            >>> print(files)
            ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', ...]
        """
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            raise ImportError(
                "huggingface_hub library is required to list tokenizer files. "
                "Install it with: pip install huggingface-hub"
            )

        # List all files in the repository
        all_files = list_repo_files(
            repo_id=model,
            revision=revision,
        )

        # Common tokenizer file patterns
        tokenizer_patterns = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "preprocessor_config.json",
            "config.json",  # Model config (may contain tokenizer info)
        ]

        # Filter files that match tokenizer patterns
        tokenizer_files = []
        for file in all_files:
            filename = Path(file).name
            # Check if file matches any tokenizer pattern
            if any(
                pattern in filename or filename == pattern
                for pattern in tokenizer_patterns
            ):
                tokenizer_files.append(file)
            # Also include files in tokenizer subdirectory if it exists
            elif file.startswith("tokenizer/"):
                tokenizer_files.append(file)

        return sorted(tokenizer_files)

    def _check_tokenizer_files_exist(self) -> tuple[bool, str | None]:
        """
        Check if tokenizer files exist in cache using direct filesystem check.

        This avoids potential network requests or hanging from huggingface_hub functions.

        Returns:
            Tuple of (files_exist: bool, cache_path: str | None)
        """
        try:
            from pathlib import Path

            # Determine cache directory
            if self.cache_dir:
                base_cache_dir = Path(self.cache_dir)
            else:
                # Use default HuggingFace cache location
                try:
                    from huggingface_hub import HF_HUB_CACHE

                    base_cache_dir = Path(HF_HUB_CACHE)
                except ImportError:
                    # Fallback to common default location
                    base_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            # HuggingFace cache structure: models--{repo_id}--{revision}/snapshots/{hash}/
            model_cache_name = self.model.replace("/", "--")
            model_cache_path = base_cache_dir / f"models--{model_cache_name}"

            if not model_cache_path.exists():
                return False, None

            # Check for snapshots directory
            snapshots_dir = model_cache_path / "snapshots"
            if not snapshots_dir.exists():
                return False, None

            # Look for any snapshot directory that contains tokenizer files
            required_files = ["tokenizer_config.json", "tokenizer.json"]

            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir():
                    # Check if this snapshot has tokenizer files
                    has_tokenizer_files = any(
                        (snapshot_dir / filename).exists()
                        for filename in required_files
                    )
                    if has_tokenizer_files:
                        return True, str(snapshot_dir)

            return False, None

        except Exception as e:
            # If filesystem check fails for any reason, fail fast
            import warnings

            warnings.warn(
                f"Failed to check local cache for model '{self.model}': {e}. "
                "Assuming files don't exist.",
                UserWarning,
            )
            return False, None

    def _ensure_model_downloaded(self) -> str:
        """
        Ensure tokenizer files are available.

        In offline mode: Check for necessary files and throw exception if not found.
        In online mode: Check files first, only download if missing.
        """
        # First, check if tokenizer files already exist in cache
        files_exist, cached_path = self._check_tokenizer_files_exist()

        if self.offline:
            # Offline mode: files must exist, throw exception if not found
            if not files_exist:
                raise OSError(
                    f"Model '{self.model}' tokenizer files not found in cache. "
                    f"Offline mode requires tokenizer files to be pre-downloaded. "
                    f"Cache dir: {self.cache_dir or 'default HuggingFace cache'}. "
                    f"Please download the model first in online mode."
                )
            # Files exist, return cached path or model ID
            return cached_path if cached_path else self.model

        # Online mode: if files exist, use them directly
        if files_exist and cached_path:
            return cached_path

        # Files don't exist or couldn't be found, need to download
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            # If huggingface_hub is not available, let AutoTokenizer handle downloads.
            return self.model

        try:
            # Download only files relevant for tokenization, ignoring model weights.
            # The returned path points to a local directory with the downloaded files.
            return snapshot_download(
                repo_id=self.model,
                cache_dir=self.cache_dir,
                revision=self.revision,
                local_files_only=False,  # Ensure we download if not present
                allow_patterns=["tokenizer*", "*.json", "*.txt", "vocab.*", "merges.*"],
                ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
            )
        except Exception as e:
            # If download fails, provide a clear error message.
            raise OSError(
                f"Failed to download tokenizer files for model '{self.model}'. "
                f"Cache dir: {self.cache_dir or 'default'}. Error: {e}"
            ) from e

    def _ensure_tokenizer(self):
        """
        Ensure tokenizer is loaded (lazy loading).

        Uses local_files_only parameter instead of environment variables for better control.
        This is the recommended approach as it doesn't affect global state.

        Raises:
            ImportError: If transformers is not available.
            OSError: If model cannot be loaded (e.g., offline mode and model not found).
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

        # Ensure model is downloaded (if needed)
        model_path = self._ensure_model_downloaded()

        # Load tokenizer
        # Since we handle all downloads ourselves in _ensure_model_downloaded(),
        # we ALWAYS use local_files_only=True for AutoTokenizer.
        # This ensures:
        # 1. No duplicate downloads
        # 2. Complete control over download process
        # 3. Consistent behavior regardless of online/offline mode
        # 4. Better error messages from our own logic
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,  # Always True - we handle downloads ourselves!
            )
        except (OSError, ValueError) as e:
            # This should rarely happen since _ensure_model_downloaded()
            # already verified files exist, but provide helpful error anyway
            mode_text = "offline" if self.offline else "online"
            raise OSError(
                f"Failed to load tokenizer for model '{self.model}' in {mode_text} mode. "
                f"Files should have been prepared by _ensure_model_downloaded(). "
                f"Cache dir: {self.cache_dir or 'default HuggingFace cache'}. "
                f"Original error: {e}"
            ) from e

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        truncation: bool | str = False,
        max_length: int | None = None,
        padding: bool | str = False,
        return_attention_mask: bool = True,
        extra: Json | None = None,
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
            OSError: If model cannot be loaded (offline mode).
        """
        # Ensure tokenizer is loaded
        self._ensure_tokenizer()

        # Normalize input to list
        is_single = isinstance(text, str)
        text_list = [text] if is_single else list(text)

        if not text_list:
            raise ValueError("Text cannot be empty")

        # Prepare tokenizer arguments
        tokenizer_kwargs: dict[str, Any] = {
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
        attention_mask = (
            encoded.get("attention_mask") if return_attention_mask else None
        )

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
