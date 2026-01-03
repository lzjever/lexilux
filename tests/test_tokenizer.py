"""
Tokenizer API client test cases
"""

import pytest
from unittest.mock import patch, MagicMock

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
            mode="force_offline",
            revision="main",
            trust_remote_code=True,
            require_transformers=True,
        )

        assert tokenizer.model == "Qwen/Qwen2.5-7B-Instruct"
        assert tokenizer.cache_dir == "/custom/cache"
        assert tokenizer.mode == "force_offline"
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


class TestTokenizerModes:
    """Tokenizer mode tests"""

    @patch("transformers.AutoTokenizer")
    def test_force_offline_mode(self, mock_auto_tokenizer):
        """Test force_offline mode"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model", mode="force_offline")
        tokenizer._ensure_tokenizer()

        # Should call with local_files_only=True
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args[1]
        assert call_kwargs["local_files_only"] is True

    @patch("transformers.AutoTokenizer")
    def test_online_mode(self, mock_auto_tokenizer):
        """Test online mode"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model", mode="online")
        tokenizer._ensure_tokenizer()

        # Should call with local_files_only=False
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args[1]
        assert call_kwargs["local_files_only"] is False

    @patch("transformers.AutoTokenizer")
    def test_auto_offline_mode_local_available(self, mock_auto_tokenizer):
        """Test auto_offline mode when local model is available"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = Tokenizer("test-model", mode="auto_offline")
        tokenizer._ensure_tokenizer()

        # Should try local first (local_files_only=True)
        calls = mock_auto_tokenizer.from_pretrained.call_args_list
        assert len(calls) == 1  # Local succeeded, no fallback
        assert calls[0][1]["local_files_only"] is True

    @patch("transformers.AutoTokenizer")
    def test_auto_offline_mode_local_unavailable(self, mock_auto_tokenizer):
        """Test auto_offline mode when local model is not available"""
        mock_tokenizer = MagicMock()

        # First call (local) raises OSError, second call (online) succeeds
        mock_auto_tokenizer.from_pretrained.side_effect = [
            OSError("Model not found locally"),
            mock_tokenizer,
        ]

        tokenizer = Tokenizer("test-model", mode="auto_offline")
        tokenizer._ensure_tokenizer()

        # Should try local first, then fallback to online
        calls = mock_auto_tokenizer.from_pretrained.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["local_files_only"] is True
        assert calls[1][1]["local_files_only"] is False

    @patch("transformers.AutoTokenizer")
    def test_force_offline_mode_failure(self, mock_auto_tokenizer):
        """Test force_offline mode when model is not found"""
        mock_auto_tokenizer.from_pretrained.side_effect = OSError("Model not found")

        tokenizer = Tokenizer("test-model", mode="force_offline")

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
