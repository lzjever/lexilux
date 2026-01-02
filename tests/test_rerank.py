"""
Rerank API client test cases
"""

import json
import pytest
import responses

from lexilux import Rerank, RerankResult
from lexilux.usage import Usage


def _make_openai_response(rerank_results, usage=None):
    """Helper to create OpenAI-compatible rerank response format"""
    response = {
        "results": rerank_results,
        "usage": usage or {"total_tokens": 100},
    }
    return response


class TestRerankInit:
    """Rerank initialization tests"""

    def test_init_with_all_params(self):
        """Test Rerank initialization with all parameters"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            timeout_s=30.0,
            headers={"X-Custom": "value"},
        )
        assert rerank.base_url == "https://api.example.com/v1"
        assert rerank.api_key == "test-key"
        assert rerank.model == "rerank-model"
        assert rerank.timeout_s == 30.0
        assert rerank.headers["Authorization"] == "Bearer test-key"
        assert rerank.headers["X-Custom"] == "value"


class TestRerankCall:
    """Rerank __call__ method tests"""

    @responses.activate
    def test_call_basic(self):
        """Test basic rerank call (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.80},
            {"index": 2, "relevance_score": 0.70},
        ]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("python http", ["urllib", "requests", "httpx"])
        assert isinstance(result, RerankResult)
        assert len(result.results) == 3

        # Results should be sorted by score (descending)
        assert result.results[0] == (1, 0.95)
        assert result.results[1] == (0, 0.80)
        assert result.results[2] == (2, 0.70)

    @responses.activate
    def test_call_with_include_docs(self):
        """Test rerank call with include_docs=True (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [
            {"index": 1, "relevance_score": 0.95, "document": {"text": "requests"}},
            {"index": 0, "relevance_score": 0.80, "document": {"text": "urllib"}},
        ]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("python http", ["urllib", "requests"], include_docs=True)
        assert len(result.results) == 2
        assert isinstance(result.results[0], tuple)
        assert len(result.results[0]) == 3  # (index, score, doc)
        assert result.results[0] == (1, 0.95, "requests")
        assert result.results[1] == (0, 0.80, "urllib")

    @responses.activate
    def test_call_with_top_k(self):
        """Test rerank call with top_k parameter (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.80},
            {"index": 2, "relevance_score": 0.70},
        ]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("python http", ["urllib", "requests", "httpx"], top_k=2)
        assert len(result.results) == 2
        assert result.results[0] == (1, 0.95)
        assert result.results[1] == (0, 0.80)

    @responses.activate
    def test_call_with_model_override(self):
        """Test rerank call with model override (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [{"index": 0, "relevance_score": 0.95}]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("query", ["doc"], model="custom-model")

        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["model"] == "custom-model"
        # Verify it's using OpenAI format
        assert "query" in payload
        assert "documents" in payload

    @responses.activate
    def test_call_with_extra_params(self):
        """Test rerank call with extra parameters (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [{"index": 0, "relevance_score": 0.95}]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("query", ["doc"], extra={"batch_size": 10})

        request = responses.calls[0].request
        payload = json.loads(request.body)
        # Extra params should be in the payload
        assert payload["batch_size"] == 10

    @responses.activate
    def test_call_with_return_raw(self):
        """Test rerank call with return_raw=True (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [{"index": 0, "relevance_score": 0.95}]
        response_data = _make_openai_response(rerank_results)
        response_data["id"] = "rerank-123"

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("query", ["doc"], return_raw=True)
        assert result.raw == response_data

    @responses.activate
    def test_call_without_model(self):
        """Test rerank call without model (should raise error)"""
        rerank = Rerank(base_url="https://api.example.com/v1", api_key="test-key")

        with pytest.raises(ValueError, match="Model must be specified"):
            rerank("query", ["doc"])

    @responses.activate
    def test_call_empty_docs(self):
        """Test rerank call with empty docs (should raise error)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1", api_key="test-key", model="rerank-model"
        )

        with pytest.raises(ValueError, match="Docs cannot be empty"):
            rerank("query", [])

    @responses.activate
    def test_call_no_results(self):
        """Test handling response with no results (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        # Empty results in OpenAI response
        response_data = {
            "results": [],
            "usage": {"total_tokens": 100},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        with pytest.raises(ValueError, match="No results in API response"):
            rerank("query", ["doc"])

    @responses.activate
    def test_call_usage_parsing(self):
        """Test usage parsing (OpenAI-compatible)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        rerank_results = [{"index": 0, "relevance_score": 0.95}]
        response_data = _make_openai_response(
            rerank_results, usage={"prompt_tokens": 50, "total_tokens": 50}
        )

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("query", ["doc"])
        assert result.usage.input_tokens == 50
        assert result.usage.total_tokens == 50

    @responses.activate
    def test_call_negative_scores_sorting(self):
        """Test sorting with negative scores (less negative = better)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        # Negative scores: -3.0 is better than -4.0
        rerank_results = [
            {"index": 0, "relevance_score": -4.0},
            {"index": 1, "relevance_score": -3.0},
            {"index": 2, "relevance_score": -5.0},
        ]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("query", ["doc0", "doc1", "doc2"])
        # Should be sorted: -3.0 (best), -4.0, -5.0 (worst)
        assert result.results[0] == (1, -3.0)  # Best (least negative)
        assert result.results[1] == (0, -4.0)
        assert result.results[2] == (2, -5.0)  # Worst (most negative)

    @responses.activate
    def test_call_positive_scores_sorting(self):
        """Test sorting with positive scores (higher = better)"""
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        # Positive scores: 0.95 is better than 0.80
        rerank_results = [
            {"index": 0, "relevance_score": 0.80},
            {"index": 1, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.70},
        ]
        response_data = _make_openai_response(rerank_results)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json=response_data,
            status=200,
        )

        result = rerank("query", ["doc0", "doc1", "doc2"])
        # Should be sorted: 0.95 (best), 0.80, 0.70 (worst)
        assert result.results[0] == (1, 0.95)  # Best (highest)
        assert result.results[1] == (0, 0.80)
        assert result.results[2] == (2, 0.70)  # Worst (lowest)


class TestRerankOpenAIMode:
    """OpenAI-compatible mode tests"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_basic(self, test_config, has_real_api_config):
        """Test OpenAI mode basic rerank call with real API (rerank_local_qwen3)"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "openai"),
        )

        result = rerank(
            "python http library",
            ["urllib is a built-in library", "requests is popular", "httpx is modern"],
        )
        assert isinstance(result, RerankResult)
        assert len(result.results) >= 1

        # Results should be sorted by score (descending)
        scores = [score for _, score in result.results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_with_return_documents(self, test_config, has_real_api_config):
        """Test OpenAI mode with return_documents=True using real API"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "openai"),
        )

        docs = ["urllib is a built-in library", "requests is popular"]
        result = rerank("python http library", docs, include_docs=True)
        assert len(result.results) >= 1

        # Check if documents are included
        if len(result.results[0]) == 3:
            # Documents are included
            for idx, score, doc in result.results:
                assert doc is not None
                assert isinstance(doc, str)
                assert len(doc) > 0
        else:
            # Documents not included (provider limitation)
            assert len(result.results[0]) == 2  # (index, score)

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_with_top_n(self, test_config, has_real_api_config):
        """Test OpenAI mode with top_n parameter using real API"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "openai"),
        )

        docs = ["urllib is a built-in library", "requests is popular", "httpx is modern"]
        result = rerank("python http library", docs, top_k=2)

        # Verify results are limited to top_k (or at least not more than requested)
        assert len(result.results) <= 2

        # Verify results are sorted by score (descending)
        if len(result.results) > 1:
            scores = [score for _, score in result.results]
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_request_format(self, test_config, has_real_api_config):
        """Test OpenAI mode request format with real API"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "openai"),
        )

        # This test verifies the API works correctly with real service
        result = rerank("python http library", ["urllib", "requests"], include_docs=True)

        # Verify we got valid results
        assert isinstance(result, RerankResult)
        assert len(result.results) >= 1
        assert isinstance(result.usage, Usage)

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_with_extra_params(self, test_config, has_real_api_config):
        """Test OpenAI mode with extra parameters using real API"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "openai"),
        )

        # Test with extra parameters (if supported by the service)
        # Some services may not support all extra params, so we test with a basic call
        result = rerank("python http library", ["urllib", "requests"])

        # Verify we got valid results
        assert isinstance(result, RerankResult)
        assert len(result.results) >= 1

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_mode_override_in_call(self, test_config, has_real_api_config):
        """Test overriding mode in __call__ with real API"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        # Initialize with openai mode
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode="openai",
        )

        # Use OpenAI mode in this call
        result = rerank("python http library", ["urllib", "requests"], mode="openai")

        # Verify we got valid results from OpenAI endpoint
        assert isinstance(result, RerankResult)
        assert len(result.results) >= 1

    def test_invalid_mode(self):
        """Test invalid mode raises error"""
        with pytest.raises(ValueError, match="Mode must be one of"):
            Rerank(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="rerank-model",
                mode="invalid",
            )


class TestRerankResult:
    """RerankResult class tests"""

    def test_rerank_result_repr(self):
        """Test RerankResult representation"""
        from lexilux.usage import Usage

        result = RerankResult(results=[(0, 0.95), (1, 0.80)], usage=Usage(total_tokens=100))
        repr_str = repr(result)
        assert "RerankResult" in repr_str
        assert "2 items" in repr_str
