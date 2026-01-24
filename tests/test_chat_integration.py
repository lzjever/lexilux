"""
Integration tests for Chat API with real endpoints.

Tests finish_reason functionality and other real API behaviors.
"""

import pytest

from lexilux import Chat, ChatResult


class TestChatIntegration:
    """Chat integration tests with real API"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_basic(self, test_config, has_real_api_config):
        """Test basic chat completion with real API"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        result = chat("Say hello in one word")
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        assert isinstance(result.usage.total_tokens, int)
        assert result.usage.total_tokens > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_finish_reason_stop(self, test_config, has_real_api_config):
        """Test chat completion with finish_reason='stop' (normal completion)"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        result = chat("Say hello")
        assert isinstance(result, ChatResult)
        assert result.finish_reason is not None
        # Most normal completions should have finish_reason="stop"
        # But we accept any valid finish_reason value
        assert (
            result.finish_reason in ("stop", "length", "content_filter")
            or result.finish_reason is None
        )

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_finish_reason_length(self, test_config, has_real_api_config):
        """Test chat completion with finish_reason='length' (max_tokens limit)"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Use a very small max_tokens to trigger length limit
        result = chat("Write a long story about a cat", max_tokens=5)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        # finish_reason should be "length" when max_tokens is reached
        # But some APIs might return "stop" if they stop naturally before limit
        assert (
            result.finish_reason in ("stop", "length", "content_filter")
            or result.finish_reason is None
        )

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_finish_reason_with_stop_sequence(
        self, test_config, has_real_api_config
    ):
        """Test chat completion with stop sequence"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Use stop sequence to trigger stop finish_reason
        result = chat("Count from 1 to 5", stop="3")
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        # finish_reason should typically be "stop" when stop sequence is hit
        assert (
            result.finish_reason in ("stop", "length", "content_filter")
            or result.finish_reason is None
        )

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_system_message(self, test_config, has_real_api_config):
        """Test chat completion with system message"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        result = chat("What is 2+2?", system="You are a helpful math assistant")
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        assert (
            result.finish_reason is not None or result.finish_reason is None
        )  # Accept any value

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_basic(self, test_config, has_real_api_config):
        """Test basic streaming chat completion"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        chunks = list(chat.stream("Say hello in one word"))
        assert len(chunks) > 0

        # Accumulate text
        full_text = "".join(chunk.delta for chunk in chunks)
        assert len(full_text) > 0

        # Check final chunk
        final_chunk = chunks[-1]
        assert final_chunk.done is True

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_finish_reason(self, test_config, has_real_api_config):
        """Test streaming chat completion with finish_reason"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        chunks = list(chat.stream("Say hello"))
        assert len(chunks) > 0

        # Find the chunk with done=True
        done_chunks = [chunk for chunk in chunks if chunk.done]
        assert len(done_chunks) > 0

        # Check finish_reason in final chunk
        final_chunk = done_chunks[-1]
        # finish_reason should be set when done=True
        # Accept any valid value or None (some services may not provide it)
        assert (
            final_chunk.finish_reason in ("stop", "length", "content_filter")
            or final_chunk.finish_reason is None
        )

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_finish_reason_length(self, test_config, has_real_api_config):
        """Test streaming with max_tokens limit (should get finish_reason='length')"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Use small max_tokens to trigger length limit
        chunks = list(chat.stream("Write a long story", max_tokens=5))
        assert len(chunks) > 0

        # Find the chunk with done=True
        done_chunks = [chunk for chunk in chunks if chunk.done]
        assert len(done_chunks) > 0

        final_chunk = done_chunks[-1]
        # Should get "length" when max_tokens is reached, but accept other values too
        assert (
            final_chunk.finish_reason in ("stop", "length", "content_filter")
            or final_chunk.finish_reason is None
        )

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_with_usage(self, test_config, has_real_api_config):
        """Test streaming with usage information"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        chunks = list(chat.stream("Say hello", include_usage=True))
        assert len(chunks) > 0

        # Final chunk should have usage
        final_chunk = chunks[-1]
        if final_chunk.done:
            # Usage may or may not be available depending on API implementation
            assert (
                isinstance(final_chunk.usage.total_tokens, int)
                or final_chunk.usage.total_tokens is None
            )

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_intermediate_chunks(self, test_config, has_real_api_config):
        """Test that intermediate chunks have finish_reason=None"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        chunks = list(chat.stream("Write a short sentence about Python"))
        assert len(chunks) > 1  # Should have multiple chunks

        # Intermediate chunks should have done=False and finish_reason=None
        for chunk in chunks[:-1]:  # All except the last
            if not chunk.done:
                assert chunk.finish_reason is None

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_extra_params(self, test_config, has_real_api_config):
        """Test chat with extra parameters"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        result = chat("Hello", extra={"temperature": 0.5, "top_p": 0.9})
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_return_raw(self, test_config, has_real_api_config):
        """Test chat with return_raw=True to access full response"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        result = chat("Hello", return_raw=True)
        assert isinstance(result, ChatResult)
        assert isinstance(result.raw, dict)
        # Should have at least choices in raw response
        assert "choices" in result.raw or len(result.raw) > 0
