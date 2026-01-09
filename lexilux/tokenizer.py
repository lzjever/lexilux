"""
Tokenizer API client (optional dependency on transformers).

Provides local tokenization with support for offline/online modes and automatic model downloading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

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
            if any(pattern in filename or filename == pattern for pattern in tokenizer_patterns):
                tokenizer_files.append(file)
            # Also include files in tokenizer subdirectory if it exists
            elif file.startswith("tokenizer/"):
                tokenizer_files.append(file)

        return sorted(tokenizer_files)

    def _ensure_model_downloaded(self) -> str:
        """
        Ensure model is downloaded to cache_dir.

        This function checks if the model is already cached locally.
        If not cached and offline=False, it downloads the model using huggingface_hub.
        The download logic is independent of AutoTokenizer.

        Returns:
            Path to the model (can be model_id or local path)

        Raises:
            OSError: If offline=True and model not found in cache, or download failed.
        """
        if self.offline:
            # Offline mode: only check cache, don't download
            return self.model

        # Online mode: check cache first, download if needed
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            # If huggingface_hub is not available, fall back to AutoTokenizer download
            return self.model

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path is None:
            # No cache_dir specified, let AutoTokenizer handle it
            return self.model

        cache_path.mkdir(parents=True, exist_ok=True)

        # Check if model is already cached (HuggingFace Hub cache structure)
        model_cache_name = self.model.replace("/", "--")
        model_cache_path = cache_path / f"models--{model_cache_name}"

        # Check for existing snapshots
        cached_snapshot_path = None
        if model_cache_path.exists():
            snapshots_dir = model_cache_path / "snapshots"
            if snapshots_dir.exists():
                # Find the first valid snapshot directory
                snapshots = sorted(snapshots_dir.iterdir())
                for snapshot in snapshots:
                    if snapshot.is_dir():
                        # Verify it's a valid snapshot (has tokenizer files)
                        tokenizer_files = list(snapshot.glob("tokenizer*.json"))
                        if tokenizer_files:
                            cached_snapshot_path = snapshot
                            break

        # Return cached path if found
        if cached_snapshot_path:
            return str(cached_snapshot_path)

        # Download only tokenizer files (not model weights)
        # Use list_tokenizer_files to identify which files to download
        try:
            from huggingface_hub import hf_hub_download

            # Get list of tokenizer files for this model
            tokenizer_files = self.list_tokenizer_files(
                self.model,
                revision=self.revision,
            )

            if not tokenizer_files:
                # Fallback: if no tokenizer files found, let AutoTokenizer handle it
                return self.model

            # Download each tokenizer file to the cache directory
            # hf_hub_download will place files in the standard HuggingFace cache structure
            # under the specified cache_dir, which AutoTokenizer can then find
            for file in tokenizer_files:
                try:
                    hf_hub_download(
                        repo_id=self.model,
                        filename=file,
                        cache_dir=str(cache_path),
                        revision=self.revision,
                        local_files_only=False,
                    )
                except Exception as e:
                    # Log warning but continue with other files
                    # Some files might not exist for all models (e.g., merges.txt for WordPiece)
                    import warnings

                    warnings.warn(
                        f"Failed to download {file}: {e}. Continuing with other files.",
                        UserWarning,
                    )

            # Return the model ID - AutoTokenizer will find the files in the HuggingFace cache
            # The files are now cached in the standard HuggingFace cache structure
            return self.model

        except ImportError:
            # If huggingface_hub functions are not available, fall back to snapshot_download
            from huggingface_hub import snapshot_download

            downloaded_path = snapshot_download(
                repo_id=self.model,
                cache_dir=str(cache_path),
                revision=self.revision,
                local_files_only=False,
                ignore_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.h5",
                    "*.ckpt",
                    "*.pb",
                    "*.onnx",
                    "model*.safetensors",
                    "pytorch_model*.bin",
                    "tf_model*.h5",
                    "flax_model*.msgpack",
                ],
            )
            return downloaded_path
        except Exception as e:
            # If download failed, raise error
            raise OSError(
                f"Failed to download model '{self.model}': {e}. Cache dir: {self.cache_dir}"
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
        # If model_path is a local path (from snapshot_download), use it directly
        # Otherwise, it's the model_id and AutoTokenizer will handle it
        if self.offline:
            # Offline: only use local cache
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    cache_dir=self.cache_dir,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True,
                )
            except (OSError, ValueError) as e:
                raise OSError(
                    f"Model '{self.model}' not found in local cache. "
                    f"Offline mode requires the model to be pre-downloaded. "
                    f"Cache dir: {self.cache_dir or 'default HuggingFace cache'}"
                ) from e
        else:
            # Online: allow network access
            # If model_path is a local snapshot path, it's already downloaded
            # If it's model_id, AutoTokenizer will download if needed
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                local_files_only=False,
            )

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
