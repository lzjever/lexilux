"""Additional tests for chat parameters"""

from lexilux.chat.params import ChatParams


class TestChatParamsAdditional:
    """Additional tests for ChatParams"""

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        params = ChatParams(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["STOP"],
            tools=[],
            tool_choice="auto",
            extra={"custom_param": "value"},
        )
        assert params.temperature == 0.7
        assert params.max_tokens == 1000
        assert params.top_p == 0.9
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.5
        assert params.stop == ["STOP"]
        assert params.extra == {"custom_param": "value"}

    def test_to_dict_excludes_none_values(self):
        """Test that None values are excluded from dict"""
        params = ChatParams(
            temperature=None,
            max_tokens=None,
        )
        result = params.to_dict()
        # None values should be excluded
        assert "temperature" not in result
        assert "max_tokens" not in result

    def test_to_dict_includes_non_none_values(self):
        """Test that non-None values are included in dict"""
        params = ChatParams(
            temperature=0.5,
            max_tokens=500,
        )
        result = params.to_dict()
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 500

    def test_to_dict_with_stop_string(self):
        """Test to_dict with stop as string"""
        params = ChatParams(stop="END")
        result = params.to_dict()
        # Stop is converted to list
        assert result["stop"] == ["END"]

    def test_to_dict_with_stop_list(self):
        """Test to_dict with stop as list"""
        params = ChatParams(stop=["STOP1", "STOP2"])
        result = params.to_dict()
        assert result["stop"] == ["STOP1", "STOP2"]

    def test_to_dict_with_extra_params(self):
        """Test that extra params are merged into dict"""
        params = ChatParams(extra={"custom": "value", "another": "param"})
        result = params.to_dict()
        assert result["custom"] == "value"
        assert result["another"] == "param"

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated"""
        params = ChatParams(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["STOP"],
            extra={"custom": "value"},
        )
        result = params.to_dict()
        assert len(result) >= 7  # At least these many fields
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
        assert result["top_p"] == 0.9
        assert result["frequency_penalty"] == 0.5
        assert result["presence_penalty"] == 0.5
        assert result["stop"] == ["STOP"]
        assert result["custom"] == "value"
