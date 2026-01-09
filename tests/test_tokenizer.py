"""
Tokenizer API client test cases
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lexilux import Tokenizer, TokenizeResult


class TestTokenizerInit:
    """Tokenizer initialization tests"""

    @patch("transformers.AutoTokenizer")
    def test_init_with_all_params(self, mock_auto_tokenizer):
        """Test Tokenizer initialization with all parameters"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer(
            "Qwen/Qwen2.5-7B-Instruct",
            cache_dir="/custom/cache",
            offline=True,
            revision="main",
            trust_remote_code=True,
            require_transformers=True,
        )

        assert tokenizer.model == "Qwen/Qwen2.5-7B-Instruct"
        assert tokenizer.cache_dir == "/custom/cache"
        assert tokenizer.offline is True
        assert tokenizer.revision == "main"
        assert tokenizer.trust_remote_code is True

    def test_init_without_transformers(self):
        """Test Tokenizer initialization without transformers (require_transformers=True)"""
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers library is required"):
                Tokenizer("test-model", require_transformers=True)

    def test_init_without_transformers_delayed(self):
        """Test Tokenizer initialization without transformers (require_transformers=False)"""
        with patch.dict("sys.modules", {"transformers": None}):
            # Should not raise error immediately
            tokenizer = Tokenizer("test-model", require_transformers=False)
            # But should raise on first use
            with pytest.raises(ImportError, match="transformers library is required"):
                tokenizer("test")

    def test_init_with_tilde_expansion(self):
        """Test Tokenizer initialization with ~ in cache_dir expands to home directory"""
        home = str(Path.home())

        # Test with just "~"
        tokenizer = Tokenizer("test-model", cache_dir="~")
        assert tokenizer.cache_dir == home

        # Test with "~/.cache/lexilux/tokenizer"
        expected_path = str(Path.home() / ".cache" / "lexilux" / "tokenizer")
        tokenizer = Tokenizer("test-model", cache_dir="~/.cache/lexilux/tokenizer")
        assert tokenizer.cache_dir == expected_path

        # Test that regular paths still work
        tokenizer = Tokenizer("test-model", cache_dir="/custom/cache")
        assert tokenizer.cache_dir == "/custom/cache"

        # Test that None still works
        tokenizer = Tokenizer("test-model", cache_dir=None)
        assert tokenizer.cache_dir is None

    @patch("huggingface_hub.list_repo_files")
    def test_list_tokenizer_files(self, mock_list_files):
        """Test list_tokenizer_files method"""
        # Mock the list_repo_files to return some files
        mock_list_files.return_value = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "config.json",
            "model.safetensors",  # Should be excluded
            "pytorch_model.bin",  # Should be excluded
            "README.md",  # Should be excluded
        ]

        files = Tokenizer.list_tokenizer_files("test-model")

        # Should only return tokenizer-related files
        assert "tokenizer.json" in files
        assert "tokenizer_config.json" in files
        assert "vocab.json" in files
        assert "merges.txt" in files
        assert "special_tokens_map.json" in files
        assert "config.json" in files
        assert "model.safetensors" not in files
        assert "pytorch_model.bin" not in files
        assert "README.md" not in files

        # Verify list_repo_files was called correctly
        mock_list_files.assert_called_once_with(
            repo_id="test-model",
            revision=None,
        )

    def test_list_tokenizer_files_without_huggingface_hub(self):
        """Test list_tokenizer_files raises ImportError when huggingface_hub is not available"""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub library is required"):
                Tokenizer.list_tokenizer_files("test-model")


class TestTokenizerModes:
    """Tokenizer mode tests"""

    @patch("transformers.AutoTokenizer")
    @patch("lexilux.tokenizer.Tokenizer._ensure_model_downloaded")
    def test_offline_mode(self, mock_ensure_download, mock_auto_tokenizer):
        """Test offline mode"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ensure_download.return_value = "test-model"

        tokenizer = Tokenizer("test-model", offline=True)
        tokenizer._ensure_tokenizer()

        # Should call with local_files_only=True
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args[1]
        assert call_kwargs["local_files_only"] is True

    @patch("transformers.AutoTokenizer")
    @patch("lexilux.tokenizer.Tokenizer._ensure_model_downloaded")
    def test_online_mode(self, mock_ensure_download, mock_auto_tokenizer):
        """Test online mode"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ensure_download.return_value = "test-model"

        tokenizer = Tokenizer("test-model", offline=False)
        tokenizer._ensure_tokenizer()

        # Should call with local_files_only=False
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args[1]
        assert call_kwargs["local_files_only"] is False

    @patch("transformers.AutoTokenizer")
    @patch("lexilux.tokenizer.Tokenizer._ensure_model_downloaded")
    def test_offline_mode_failure(self, mock_ensure_download, mock_auto_tokenizer):
        """Test offline mode when model is not found"""
        mock_ensure_download.return_value = "test-model"
        mock_auto_tokenizer.from_pretrained.side_effect = OSError("Model not found")

        tokenizer = Tokenizer("test-model", offline=True)

        with pytest.raises(OSError, match="not found in local cache"):
            tokenizer._ensure_tokenizer()


class TestTokenizerCall:
    """Tokenizer __call__ method tests"""

    @patch("transformers.AutoTokenizer")
    def test_call_with_single_string(self, mock_auto_tokenizer):
        """Test calling tokenizer with a single string"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[15496, 11, 1917, 0]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model")
        result = tokenizer("Hello, world!")

        assert isinstance(result, TokenizeResult)
        assert result.input_ids == [[15496, 11, 1917, 0]]
        assert result.attention_mask == [[1, 1, 1, 1]]
        assert result.usage.input_tokens == 4
        assert result.usage.total_tokens == 4

    @patch("transformers.AutoTokenizer")
    def test_call_with_list(self, mock_auto_tokenizer):
        """Test calling tokenizer with a list of strings"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[15496, 0], [1917, 0]],
            "attention_mask": [[1, 1], [1, 1]],
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model")
        result = tokenizer(["Hello", "world"])

        assert len(result.input_ids) == 2
        assert result.input_ids[0] == [15496, 0]
        assert result.input_ids[1] == [1917, 0]
        assert result.usage.input_tokens == 4  # 2 + 2

    @patch("transformers.AutoTokenizer")
    def test_call_with_parameters(self, mock_auto_tokenizer):
        """Test calling tokenizer with additional parameters"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[15496, 0]],
            "attention_mask": [[1, 1]],
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model")
        tokenizer(
            "Hello",
            add_special_tokens=False,
            truncation=True,
            max_length=10,
            padding="max_length",
        )

        # Verify tokenizer was called with correct parameters
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["add_special_tokens"] is False
        assert call_kwargs["truncation"] is True
        assert call_kwargs["max_length"] == 10
        assert call_kwargs["padding"] == "max_length"

    @patch("transformers.AutoTokenizer")
    def test_call_without_attention_mask(self, mock_auto_tokenizer):
        """Test calling tokenizer without attention mask"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[15496, 0]],
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model")
        result = tokenizer("Hello", return_attention_mask=False)

        assert result.attention_mask is None

    @patch("transformers.AutoTokenizer")
    def test_call_with_return_raw(self, mock_auto_tokenizer):
        """Test calling tokenizer with return_raw=True"""
        mock_tokenizer = MagicMock()
        raw_output = {
            "input_ids": [[15496, 0]],
            "attention_mask": [[1, 1]],
            "token_type_ids": [[0, 0]],
        }
        mock_tokenizer.return_value = raw_output
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model")
        result = tokenizer("Hello", return_raw=True)

        assert result.raw == raw_output

    @patch("transformers.AutoTokenizer")
    def test_call_empty_input(self, mock_auto_tokenizer):
        """Test calling tokenizer with empty input"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            tokenizer([])


class TestTokenizeResult:
    """TokenizeResult class tests"""

    def test_tokenize_result_repr(self):
        """Test TokenizeResult representation"""
        from lexilux.usage import Usage

        result = TokenizeResult(
            input_ids=[[1, 2, 3], [4, 5]],
            attention_mask=[[1, 1, 1], [1, 1]],
            usage=Usage(input_tokens=5),
        )
        repr_str = repr(result)
        assert "TokenizeResult" in repr_str
        assert "2 sequences" in repr_str
