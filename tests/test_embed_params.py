"""Test embed parameters"""

from lexilux.embed_params import EmbedParams


class TestEmbedParams:
    """Test EmbedParams dataclass"""

    def test_init_with_default_values(self):
        """Test initialization with default values"""
        params = EmbedParams()
        assert params.dimensions is None
        assert params.encoding_format == "float"
        assert params.user is None
        assert params.extra is None

    def test_init_with_all_values(self):
        """Test initialization with all values"""
        params = EmbedParams(
            dimensions=512,
            encoding_format="base64",
            user="user123",
            extra={"custom": "value"},
        )
        assert params.dimensions == 512
        assert params.encoding_format == "base64"
        assert params.user == "user123"
        assert params.extra == {"custom": "value"}

    def test_to_dict_with_none_values_excluded(self):
        """Test that None values are excluded from dict by default"""
        params = EmbedParams()
        result = params.to_dict()
        # encoding_format has default value "float", but it's excluded because it equals default
        assert "dimensions" not in result
        assert "user" not in result
        assert "extra" not in result

    def test_to_dict_with_all_values(self):
        """Test to_dict with all values"""
        params = EmbedParams(
            dimensions=512,
            encoding_format="float",
            user="user123",
            extra={"custom": "value"},
        )
        result = params.to_dict()
        assert (
            len(result) == 3
        )  # dimensions, user, custom (encoding_format excluded as default)
        assert result["dimensions"] == 512
        assert result["user"] == "user123"
        assert result["custom"] == "value"

    def test_to_dict_with_extra_parameters(self):
        """Test that extra parameters are merged into dict"""
        params = EmbedParams(extra={"param1": "value1", "param2": "value2"})
        result = params.to_dict()
        assert result["param1"] == "value1"
        assert result["param2"] == "value2"

    def test_to_dict_exclude_none_false(self):
        """Test to_dict with exclude_none=False"""
        params = EmbedParams(dimensions=None, user=None)
        result = params.to_dict(exclude_none=False)
        # Even with exclude_none=False, None values are not added
        assert "dimensions" not in result
        assert "user" not in result

    def test_to_dict_with_non_default_encoding_format(self):
        """Test to_dict with non-default encoding_format"""
        params = EmbedParams(encoding_format="base64")
        result = params.to_dict()
        assert "encoding_format" in result
        assert result["encoding_format"] == "base64"
