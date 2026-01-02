"""
Usage and ResultBase test cases
"""

from lexilux.usage import Usage, ResultBase, Json


class TestUsage:
    """Usage class tests"""

    def test_create_usage_with_all_fields(self):
        """Test creating Usage with all fields"""
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            details={"cached_tokens": 5},
        )
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.total_tokens == 30
        assert usage.details == {"cached_tokens": 5}

    def test_create_usage_with_partial_fields(self):
        """Test creating Usage with partial fields"""
        usage = Usage(input_tokens=10)
        assert usage.input_tokens == 10
        assert usage.output_tokens is None
        assert usage.total_tokens is None
        assert usage.details == {}

    def test_create_empty_usage(self):
        """Test creating empty Usage"""
        usage = Usage()
        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None
        assert usage.details == {}

    def test_usage_repr(self):
        """Test Usage string representation"""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        repr_str = repr(usage)
        assert "input_tokens=10" in repr_str
        assert "output_tokens=20" in repr_str
        assert "total_tokens=30" in repr_str

    def test_usage_equality(self):
        """Test Usage equality"""
        usage1 = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        usage2 = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        usage3 = Usage(input_tokens=10, output_tokens=20, total_tokens=40)

        assert usage1 == usage2
        assert usage1 != usage3

    def test_usage_with_details(self):
        """Test Usage with details dictionary"""
        details = {"cached_tokens": 5, "reasoning_tokens": 10}
        usage = Usage(total_tokens=100, details=details)
        assert usage.details == details
        assert usage.details["cached_tokens"] == 5


class TestResultBase:
    """ResultBase class tests"""

    def test_create_result_base(self):
        """Test creating ResultBase"""
        usage = Usage(total_tokens=10)
        result = ResultBase(usage=usage)
        assert result.usage == usage
        assert result.raw == {}

    def test_create_result_base_with_raw(self):
        """Test creating ResultBase with raw data"""
        usage = Usage(total_tokens=10)
        raw_data = {"id": "test", "model": "gpt-4"}
        result = ResultBase(usage=usage, raw=raw_data)
        assert result.usage == usage
        assert result.raw == raw_data

    def test_result_base_repr(self):
        """Test ResultBase string representation"""
        usage = Usage(total_tokens=10)
        result = ResultBase(usage=usage, raw={"test": "data"})
        repr_str = repr(result)
        assert "ResultBase" in repr_str
        assert "usage=" in repr_str
        assert "raw=" in repr_str

    def test_result_base_inheritance(self):
        """Test that ResultBase can be subclassed"""
        usage = Usage(total_tokens=10)

        class CustomResult(ResultBase):
            def __init__(self, *, usage: Usage, value: str, raw: Json | None = None):
                super().__init__(usage=usage, raw=raw)
                self.value = value

        result = CustomResult(usage=usage, value="test")
        assert result.usage == usage
        assert result.value == "test"
        assert isinstance(result, ResultBase)
