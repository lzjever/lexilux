"""
Comprehensive tests for all three rerank modes with real APIs.

Tests OpenAI (Jina), DashScope, and Chat modes individually and verifies
consistent behavior across all providers.
"""

import pytest

from lexilux import Rerank, RerankResult
from lexilux.usage import Usage


class TestRerankOpenAIModeReal:
    """OpenAI-compatible mode tests with real Jina API"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_jina_basic(self, test_config, has_real_api_config):
        """Test OpenAI mode with rerank_local_qwen3"""
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
            [
                "urllib is a built-in Python library for HTTP requests",
                "requests is a popular third-party HTTP library for Python",
                "httpx is a modern async HTTP client for Python",
            ],
        )
        assert isinstance(result, RerankResult)
        assert len(result.results) == 3

        # Results should be sorted by score (descending)
        scores = [score for _, score in result.results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_jina_with_documents(self, test_config, has_real_api_config):
        """Test OpenAI mode with rerank_local_qwen3 and include_docs=True"""
        if not has_real_api_config or "rerank_local_qwen3" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_local_qwen3"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "openai"),
        )

        docs = ["urllib is a built-in Python library", "requests is a popular third-party library"]
        result = rerank("python http library", docs, include_docs=True)
        assert len(result.results) >= 1
        assert isinstance(result.results[0], tuple)
        assert len(result.results[0]) == 3  # (index, score, doc)

        # Verify document text is included
        for idx, score, doc in result.results:
            assert doc is not None
            assert isinstance(doc, str)
            assert len(doc) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_openai_mode_jina_with_top_k(self, test_config, has_real_api_config):
        """Test OpenAI mode with rerank_local_qwen3 and top_k parameter"""
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

        # Verify results are limited to top_k
        assert len(result.results) == 2

        # Verify results are sorted by score (descending)
        scores = [score for _, score in result.results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestRerankDashScopeModeReal:
    """DashScope mode tests with real Alibaba Cloud API"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_dashscope_mode_basic(self, test_config, has_real_api_config):
        """Test DashScope mode basic rerank call"""
        if not has_real_api_config or "rerank_dashscope" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_dashscope"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "dashscope"),
        )

        result = rerank(
            "python http library",
            [
                "urllib is a built-in Python library for HTTP requests",
                "requests is a popular third-party HTTP library for Python",
                "httpx is a modern async HTTP client for Python",
            ],
        )
        assert isinstance(result, RerankResult)
        assert len(result.results) >= 1

        # Results should be sorted by score (descending)
        scores = [score for _, score in result.results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_dashscope_mode_with_top_k(self, test_config, has_real_api_config):
        """Test DashScope mode with top_k parameter"""
        if not has_real_api_config or "rerank_dashscope" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["rerank_dashscope"]
        rerank = Rerank(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
            mode=config.get("mode", "dashscope"),
        )

        docs = ["urllib is a built-in library", "requests is popular", "httpx is modern"]
        result = rerank("python http library", docs, top_k=2)

        # Verify results are limited to top_k
        assert len(result.results) <= 2

        # Verify results are sorted by score (descending)
        if len(result.results) > 1:
            scores = [score for _, score in result.results]
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
