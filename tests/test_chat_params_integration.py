"""
Integration tests for ChatParams with real endpoints.

Tests ChatParams dataclass functionality and parameter passing with real API.
Note: Some parameters may not be supported by all OpenAI-compatible servers,
which is expected behavior. We test that parameters are correctly passed
to the API, not that all servers support all parameters.
"""

import pytest

from lexilux import Chat, ChatParams, ChatResult


class TestChatParamsIntegration:
    """ChatParams integration tests with real API"""

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_basic(self, test_config, has_real_api_config):
        """Test basic chat completion using ChatParams"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(temperature=0.5, max_tokens=50)
        result = chat("Say hello", params=params)

        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        assert isinstance(result.usage.total_tokens, int)
        assert result.usage.total_tokens > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_temperature(self, test_config, has_real_api_config):
        """Test ChatParams with temperature parameter"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Test with low temperature (more deterministic)
        params = ChatParams(temperature=0.1)
        result1 = chat("Say hello", params=params)
        assert isinstance(result1, ChatResult)
        assert len(result1.text) > 0

        # Test with high temperature (more random)
        params = ChatParams(temperature=1.5)
        result2 = chat("Say hello", params=params)
        assert isinstance(result2, ChatResult)
        assert len(result2.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_top_p(self, test_config, has_real_api_config):
        """Test ChatParams with top_p parameter"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(top_p=0.9)
        result = chat("Say hello", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_max_tokens(self, test_config, has_real_api_config):
        """Test ChatParams with max_tokens parameter"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Use small max_tokens to test limit
        params = ChatParams(max_tokens=10)
        result = chat("Write a long story", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        # Result should be limited by max_tokens
        # finish_reason might be "length" if limit is reached

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_stop(self, test_config, has_real_api_config):
        """Test ChatParams with stop sequences"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Test with single stop string
        params = ChatParams(stop=".")
        result1 = chat("Count from 1 to 5", params=params)
        assert isinstance(result1, ChatResult)
        assert len(result1.text) > 0

        # Test with list of stop strings
        params = ChatParams(stop=["3", "4"])
        result2 = chat("Count from 1 to 5", params=params)
        assert isinstance(result2, ChatResult)
        assert len(result2.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_presence_penalty(
        self, test_config, has_real_api_config
    ):
        """Test ChatParams with presence_penalty parameter"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(presence_penalty=0.5)
        result = chat("Tell me about Python", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_frequency_penalty(
        self, test_config, has_real_api_config
    ):
        """Test ChatParams with frequency_penalty parameter"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(frequency_penalty=0.5)
        result = chat("Tell me about Python", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_user(self, test_config, has_real_api_config):
        """Test ChatParams with user parameter"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(user="test-user-123")
        result = chat("Say hello", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_combined(self, test_config, has_real_api_config):
        """Test ChatParams with multiple parameters combined"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,
            presence_penalty=0.2,
            frequency_penalty=0.1,
            user="test-user",
        )
        result = chat("Tell me a joke", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_params_override_with_direct_args(
        self, test_config, has_real_api_config
    ):
        """Test that direct arguments override ChatParams values"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # ChatParams has temperature=0.5, but direct arg should override to 0.9
        params = ChatParams(temperature=0.5, max_tokens=30)
        result = chat("Say hello", params=params, temperature=0.9)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_with_chatparams_extra(self, test_config, has_real_api_config):
        """Test ChatParams with extra custom parameters"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Test with extra parameters in ChatParams
        params = ChatParams(
            temperature=0.5,
            extra={"custom_param": "value", "another_param": 123},
        )
        result = chat("Say hello", params=params)
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_params_with_extra_parameter(self, test_config, has_real_api_config):
        """Test ChatParams combined with extra parameter in chat call"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(temperature=0.5)
        # extra parameter should be merged with params
        result = chat("Say hello", params=params, extra={"custom": "value"})
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_with_chatparams(self, test_config, has_real_api_config):
        """Test streaming chat completion with ChatParams"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(temperature=0.5, max_tokens=30)
        chunks = list(chat.stream("Say hello", params=params))
        assert len(chunks) > 0

        # Accumulate text
        full_text = "".join(chunk.delta for chunk in chunks)
        assert len(full_text) > 0

        # Check final chunk
        final_chunk = chunks[-1]
        assert final_chunk.done is True

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_stream_with_chatparams_stop(self, test_config, has_real_api_config):
        """Test streaming with ChatParams stop sequences"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        params = ChatParams(stop=".")
        chunks = list(chat.stream("Count from 1 to 5", params=params))
        assert len(chunks) > 0

        full_text = "".join(chunk.delta for chunk in chunks)
        assert len(full_text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_backward_compatibility(self, test_config, has_real_api_config):
        """Test that direct parameter passing still works (backward compatibility)"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        config = test_config["completion"]
        chat = Chat(
            base_url=config["api_base"],
            api_key=config["api_key"],
            model=config["model"],
        )

        # Test old-style direct parameter passing (should still work)
        result = chat(
            "Say hello",
            temperature=0.5,
            max_tokens=30,
            stop=".",
        )
        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    @pytest.mark.integration
    @pytest.mark.skip_if_no_config
    def test_chat_params_to_dict(self, test_config, has_real_api_config):
        """Test ChatParams.to_dict() method"""
        if not has_real_api_config or "completion" not in test_config:
            pytest.skip("No real API config available")

        # Test to_dict with all parameters
        params = ChatParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["stop1", "stop2"],
            presence_penalty=0.5,
            frequency_penalty=0.3,
            user="test-user",
            n=2,
        )
        param_dict = params.to_dict()

        assert param_dict["temperature"] == 0.7
        assert param_dict["top_p"] == 0.9
        assert param_dict["max_tokens"] == 100
        assert param_dict["stop"] == ["stop1", "stop2"]
        assert param_dict["presence_penalty"] == 0.5
        assert param_dict["frequency_penalty"] == 0.3
        assert param_dict["user"] == "test-user"
        assert param_dict["n"] == 2

        # Test to_dict with exclude_none=False
        params2 = ChatParams(temperature=0.5)
        param_dict2 = params2.to_dict(exclude_none=False)
        assert "temperature" in param_dict2
        assert "max_tokens" in param_dict2  # Should include None values

        # Test to_dict with extra parameters
        params3 = ChatParams(temperature=0.5, extra={"custom": "value"})
        param_dict3 = params3.to_dict()
        assert param_dict3["temperature"] == 0.5
        assert param_dict3["custom"] == "value"
