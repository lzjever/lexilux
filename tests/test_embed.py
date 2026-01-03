"""
Embedding API client test cases
"""

import json
import pytest
import responses

from lexilux import Embed, EmbedResult
from lexilux.usage import Usage


class TestEmbedInit:
    """Embed initialization tests"""

    def test_init_with_all_params(self):
        """Test Embed initialization with all parameters"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
            timeout_s=30.0,
            headers={"X-Custom": "value"},
        )
        assert embed.base_url == "https://api.example.com/v1"
        assert embed.api_key == "test-key"
        assert embed.model == "text-embedding-ada-002"
        assert embed.timeout_s == 30.0
        assert embed.headers["Authorization"] == "Bearer test-key"
        assert embed.headers["X-Custom"] == "value"


class TestEmbedCall:
    """Embed __call__ method tests"""

    @responses.activate
    def test_call_with_single_string(self):
        """Test calling embed with a single string"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=response_data,
            status=200,
        )

        result = embed("Hello")
        assert isinstance(result, EmbedResult)
        assert isinstance(result.vectors, list)
        assert len(result.vectors) == 5
        assert result.vectors == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result.usage.total_tokens == 5

    @responses.activate
    def test_call_with_list(self):
        """Test calling embed with a list of strings"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=response_data,
            status=200,
        )

        result = embed(["Hello", "World"])
        assert isinstance(result, EmbedResult)
        assert isinstance(result.vectors, list)
        assert len(result.vectors) == 2
        assert result.vectors[0] == [0.1, 0.2, 0.3]
        assert result.vectors[1] == [0.4, 0.5, 0.6]

    @responses.activate
    def test_call_with_tuple(self):
        """Test calling embed with a tuple"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        response_data = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "usage": {"total_tokens": 5},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=response_data,
            status=200,
        )

        result = embed(("Hello",))
        assert isinstance(result.vectors, list)
        # Tuple is treated as iterable, so it becomes a batch of 1 item
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == 2  # Vector dimension

    @responses.activate
    def test_call_with_model_override(self):
        """Test calling embed with model override"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        response_data = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "usage": {"total_tokens": 5},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=response_data,
            status=200,
        )

        embed("Hello", model="text-embedding-3-large")

        # Verify request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["model"] == "text-embedding-3-large"

    @responses.activate
    def test_call_with_extra_params(self):
        """Test calling embed with extra parameters"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        response_data = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "usage": {"total_tokens": 5},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=response_data,
            status=200,
        )

        embed("Hello", extra={"encoding_format": "base64"})

        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["encoding_format"] == "base64"

    @responses.activate
    def test_call_with_return_raw(self):
        """Test calling embed with return_raw=True"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        response_data = {
            "id": "embed-123",
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "usage": {"total_tokens": 5},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=response_data,
            status=200,
        )

        result = embed("Hello", return_raw=True)
        assert result.raw == response_data

    @responses.activate
    def test_call_without_model(self):
        """Test calling embed without model (should raise error)"""
        embed = Embed(base_url="https://api.example.com/v1", api_key="test-key")

        with pytest.raises(ValueError, match="Model must be specified"):
            embed("Hello")

    @responses.activate
    def test_call_empty_input(self):
        """Test calling embed with empty input (should raise error)"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        with pytest.raises(ValueError, match="Input cannot be empty"):
            embed([])

    @responses.activate
    def test_call_no_data(self):
        """Test handling response with no data"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json={"data": []},
            status=200,
        )

        with pytest.raises(ValueError, match="No data in API response"):
            embed("Hello")

    @responses.activate
    def test_call_usage_parsing(self):
        """Test usage parsing from different response formats"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        # Test OpenAI-style usage
        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json={
                "data": [{"embedding": [0.1, 0.2], "index": 0}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
            status=200,
        )

        result = embed("Hello")
        assert result.usage.input_tokens == 5
        assert result.usage.total_tokens == 5

        # Test alternative usage format
        responses.reset()
        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json={
                "data": [{"embedding": [0.1, 0.2], "index": 0}],
                "usage": {"input_tokens": 10, "total_tokens": 10},
            },
            status=200,
        )

        result = embed("Hello")
        assert result.usage.input_tokens == 10
        assert result.usage.total_tokens == 10


class TestEmbedResult:
    """EmbedResult class tests"""

    def test_embed_result_repr_single(self):
        """Test EmbedResult representation for single vector"""
        from lexilux.usage import Usage

        result = EmbedResult(vectors=[0.1, 0.2, 0.3], usage=Usage(total_tokens=5))
        repr_str = repr(result)
        assert "EmbedResult" in repr_str
        assert "3 dims" in repr_str

    def test_embed_result_repr_multiple(self):
        """Test EmbedResult representation for multiple vectors"""
        from lexilux.usage import Usage

        result = EmbedResult(vectors=[[0.1, 0.2], [0.3, 0.4]], usage=Usage(total_tokens=10))
        repr_str = repr(result)
        assert "EmbedResult" in repr_str
        assert "2 vectors" in repr_str


class TestEmbedRealAPI:
    """Real API embedding tests"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_embed_real_api_basic(self, test_config, has_real_api_config):
        """Test embedding with real API (embed_local_qwen3)"""
        if not has_real_api_config or "embed_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["embed_local_qwen3"]
        embed = Embed(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        result = embed("Hello, world!")
        assert isinstance(result, EmbedResult)
        assert isinstance(result.vectors, list)
        assert len(result.vectors) > 0  # Should have embedding vector
        assert isinstance(result.vectors[0], (int, float))  # Single vector case
        assert isinstance(result.usage, Usage)

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_embed_real_api_multiple(self, test_config, has_real_api_config):
        """Test embedding multiple texts with real API"""
        if not has_real_api_config or "embed_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["embed_local_qwen3"]
        embed = Embed(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        texts = ["Hello", "World", "Python"]
        result = embed(texts)
        assert isinstance(result, EmbedResult)
        assert isinstance(result.vectors, list)
        assert len(result.vectors) == 3  # Should have 3 vectors
        assert all(isinstance(v, list) for v in result.vectors)  # Each should be a vector
        assert all(len(v) > 0 for v in result.vectors)  # Each vector should have dimensions
        assert isinstance(result.usage, Usage)

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_embed_real_api_with_model_override(self, test_config, has_real_api_config):
        """Test embedding with model override using real API"""
        if not has_real_api_config or "embed_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["embed_local_qwen3"]
        embed = Embed(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Use the same model (override should work)
        result = embed("Test", model=config["model"])
        assert isinstance(result, EmbedResult)
        assert len(result.vectors) > 0
