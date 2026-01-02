"""
Rerank mode consistency tests.

Tests that all two rerank modes (OpenAI, DashScope) produce consistent
RerankResult behavior, hiding provider differences from users.
"""

import pytest

from lexilux import Rerank, RerankResult
from lexilux.usage import Usage


class TestRerankModesConsistency:
    """Test that all rerank modes produce consistent results"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_all_modes_basic_consistency(self, test_config, has_real_api_config):
        """Test that all two modes return consistent result structure"""
        if not has_real_api_config:
            pytest.skip("No real API config available")

        query = "python http library"
        docs = [
            "urllib is a built-in Python library for HTTP requests",
            "requests is a popular third-party HTTP library for Python",
            "httpx is a modern async HTTP client for Python",
        ]

        results_by_mode = {}

        # Test OpenAI mode (rerank_local_qwen3)
        if "rerank_local_qwen3" in test_config:
            config = test_config["rerank_local_qwen3"]
            rerank = Rerank(
                base_url=config["api_base"],
                api_key=config["api_key"],
                model=config["model"],
                mode=config.get("mode", "openai"),
            )
            result = rerank(query, docs)
            results_by_mode["openai"] = result
            assert isinstance(result, RerankResult)
            assert len(result.results) >= 1
            assert isinstance(result.usage, Usage)

        # Test DashScope mode
        if "rerank_dashscope" in test_config:
            config = test_config["rerank_dashscope"]
            rerank = Rerank(
                base_url=config["api_base"],
                api_key=config["api_key"],
                model=config["model"],
                mode=config.get("mode", "dashscope"),
            )
            result = rerank(query, docs)
            results_by_mode["dashscope"] = result
            assert isinstance(result, RerankResult)
            assert len(result.results) >= 1
            assert isinstance(result.usage, Usage)

        # Verify all modes return consistent structure
        if len(results_by_mode) > 1:
            # All results should have the same structure
            first_result = list(results_by_mode.values())[0]
            for mode, result in results_by_mode.items():
                assert isinstance(result, RerankResult), f"{mode} mode should return RerankResult"
                assert len(result.results) == len(
                    first_result.results
                ), f"{mode} mode should return same number of results"
                assert isinstance(result.results[0], tuple), f"{mode} mode results should be tuples"
                assert (
                    len(result.results[0]) == 2
                ), f"{mode} mode results should be (index, score) tuples"
                assert isinstance(result.usage, Usage), f"{mode} mode should return Usage object"

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_all_modes_score_sorting_consistency(self, test_config, has_real_api_config):
        """Test that all modes sort scores consistently (descending)"""
        if not has_real_api_config:
            pytest.skip("No real API config available")

        query = "machine learning algorithms"
        docs = [
            "Neural networks are computational models inspired by biological neural networks",
            "Support vector machines are supervised learning models for classification",
            "Random forests combine multiple decision trees for better accuracy",
        ]

        results_by_mode = {}

        # Test each available mode
        for config_key, mode_name in [
            ("rerank_local_qwen3", "openai"),
            ("rerank_dashscope", "dashscope"),
        ]:
            if config_key in test_config:
                config = test_config[config_key]
                rerank = Rerank(
                    base_url=config["api_base"],
                    api_key=config["api_key"],
                    model=config["model"],
                    mode=config.get("mode", mode_name),
                )
                result = rerank(query, docs)
                results_by_mode[mode_name] = result

        # Verify all modes sort scores in descending order
        for mode, result in results_by_mode.items():
            if len(result.results) > 1:
                scores = [score for _, score in result.results]
                # Scores should be sorted descending (higher is better)
                assert all(
                    scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
                ), f"{mode} mode should sort scores in descending order"

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_all_modes_top_k_consistency(self, test_config, has_real_api_config):
        """Test that all modes respect top_k parameter consistently"""
        if not has_real_api_config:
            pytest.skip("No real API config available")

        query = "python programming"
        docs = [
            "Python is a high-level programming language",
            "Java is an object-oriented programming language",
            "JavaScript is a scripting language for web development",
            "C++ is a compiled programming language",
            "Go is a statically typed compiled language",
        ]

        top_k = 2
        results_by_mode = {}

        # Test each available mode
        for config_key, mode_name in [
            ("rerank_local_qwen3", "openai"),
            ("rerank_dashscope", "dashscope"),
        ]:
            if config_key in test_config:
                config = test_config[config_key]
                rerank = Rerank(
                    base_url=config["api_base"],
                    api_key=config["api_key"],
                    model=config["model"],
                    mode=config.get("mode", mode_name),
                )
                result = rerank(query, docs, top_k=top_k)
                results_by_mode[mode_name] = result

        # Verify all modes respect top_k
        for mode, result in results_by_mode.items():
            assert (
                len(result.results) <= top_k
            ), f"{mode} mode should return at most {top_k} results"

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_all_modes_include_docs_consistency(self, test_config, has_real_api_config):
        """Test that all modes handle include_docs consistently"""
        if not has_real_api_config:
            pytest.skip("No real API config available")

        query = "data science"
        docs = [
            "Data science combines statistics and programming",
            "Machine learning is a subset of artificial intelligence",
        ]

        results_by_mode = {}

        # Test each available mode with include_docs=True
        for config_key, mode_name in [
            ("rerank_local_qwen3", "openai"),
            ("rerank_dashscope", "dashscope"),
        ]:
            if config_key in test_config:
                config = test_config[config_key]
                rerank = Rerank(
                    base_url=config["api_base"],
                    api_key=config["api_key"],
                    model=config["model"],
                    mode=config.get("mode", mode_name),
                )
                result = rerank(query, docs, include_docs=True)
                results_by_mode[mode_name] = result

        # Verify all modes return consistent structure when include_docs=True
        for mode, result in results_by_mode.items():
            assert len(result.results) >= 1, f"{mode} mode should return results"
            # Some providers may not return documents, so we check what we get
            if len(result.results[0]) == 3:
                # Documents are included
                for idx, score, doc in result.results:
                    assert doc is not None, f"{mode} mode should include document text"
                    assert isinstance(doc, str), f"{mode} mode document should be string"
            else:
                # Documents not included (provider limitation)
                assert (
                    len(result.results[0]) == 2
                ), f"{mode} mode should return (index, score) when docs not available"

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_all_modes_usage_consistency(self, test_config, has_real_api_config):
        """Test that all modes return usage information consistently"""
        if not has_real_api_config:
            pytest.skip("No real API config available")

        query = "natural language processing"
        docs = ["NLP is a field of AI", "Text analysis uses NLP techniques"]

        results_by_mode = {}

        # Test each available mode
        for config_key, mode_name in [
            ("rerank_local_qwen3", "openai"),
            ("rerank_dashscope", "dashscope"),
        ]:
            if config_key in test_config:
                config = test_config[config_key]
                rerank = Rerank(
                    base_url=config["api_base"],
                    api_key=config["api_key"],
                    model=config["model"],
                    mode=config.get("mode", mode_name),
                )
                result = rerank(query, docs)
                results_by_mode[mode_name] = result

        # Verify all modes return Usage object
        for mode, result in results_by_mode.items():
            assert isinstance(result.usage, Usage), f"{mode} mode should return Usage object"
            # Usage may or may not have token counts (provider-dependent)
            # But the structure should be consistent

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_all_modes_index_mapping_consistency(self, test_config, has_real_api_config):
        """Test that all modes correctly map results to original document indices"""
        if not has_real_api_config:
            pytest.skip("No real API config available")

        query = "web framework"
        docs = [
            "Django is a Python web framework",
            "Flask is a lightweight Python web framework",
            "FastAPI is a modern Python web framework",
        ]

        results_by_mode = {}

        # Test each available mode
        for config_key, mode_name in [
            ("rerank_local_qwen3", "openai"),
            ("rerank_dashscope", "dashscope"),
        ]:
            if config_key in test_config:
                config = test_config[config_key]
                rerank = Rerank(
                    base_url=config["api_base"],
                    api_key=config["api_key"],
                    model=config["model"],
                    mode=config.get("mode", mode_name),
                )
                result = rerank(query, docs)
                results_by_mode[mode_name] = result

        # Verify all modes return valid indices
        for mode, result in results_by_mode.items():
            for idx, score in result.results:
                assert isinstance(idx, int), f"{mode} mode should return integer indices"
                assert (
                    0 <= idx < len(docs)
                ), f"{mode} mode index {idx} should be in range [0, {len(docs)})"
                assert isinstance(score, (int, float)), f"{mode} mode should return numeric scores"
