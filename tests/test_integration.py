"""
Integration tests for lexilux
"""

import responses

from lexilux import Chat, Embed, Rerank, Usage


class TestIntegration:
    """Integration tests"""

    @responses.activate
    def test_chat_basic_flow(self):
        """Test basic chat flow"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello!"}}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        result = chat("Hi")
        assert result.text == "Hello!"
        assert isinstance(result.usage, Usage)
        assert result.usage.total_tokens == 10

    @responses.activate
    def test_embed_basic_flow(self):
        """Test basic embed flow"""
        embed = Embed(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
        )

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json={
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                "usage": {"total_tokens": 5},
            },
            status=200,
        )

        result = embed("Hello")
        assert len(result.vectors) == 3
        assert isinstance(result.usage, Usage)

    @responses.activate
    def test_rerank_basic_flow(self):
        """Test basic rerank flow"""
        # Use OpenAI mode for this test to match the /rerank endpoint
        rerank = Rerank(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="rerank-model",
            mode="openai",
        )

        responses.add(
            responses.POST,
            "https://api.example.com/v1/rerank",
            json={
                "results": [{"index": 0, "relevance_score": 0.95}],
                "usage": {"total_tokens": 50},
            },
            status=200,
        )

        result = rerank("query", ["doc1", "doc2"])
        assert len(result.results) == 1
        assert isinstance(result.usage, Usage)
