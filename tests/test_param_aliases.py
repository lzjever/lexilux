"""Test parameter alias mapping functionality."""

import pytest
from lexilux.chat.params import ChatParams


def test_param_aliases_basic():
    """Test basic parameter alias mapping."""
    params = ChatParams(
        temperature=0.8,
        top_p=0.9,
        param_aliases={"temperature": "temp", "top_p": "nucleus_sampling"},
    )

    result = params.to_dict()

    # Check aliases are applied
    assert "temp" in result
    assert "nucleus_sampling" in result
    assert result["temp"] == 0.8
    assert result["nucleus_sampling"] == 0.9

    # Check original keys are removed
    assert "temperature" not in result
    assert "top_p" not in result


def test_param_aliases_partial():
    """Test partial parameter alias mapping."""
    params = ChatParams(
        temperature=0.7,
        max_tokens=100,
        param_aliases={
            "temperature": "temp"  # Only alias temperature, not max_tokens
        },
    )

    result = params.to_dict()

    # Temperature should be aliased
    assert "temp" in result
    assert result["temp"] == 0.7
    assert "temperature" not in result

    # max_tokens should remain unchanged
    assert "max_tokens" in result
    assert result["max_tokens"] == 100


def test_param_aliases_none():
    """Test parameters without aliases."""
    params = ChatParams(temperature=0.7, max_tokens=100, param_aliases=None)

    result = params.to_dict()

    # All parameters should keep original names
    assert "temperature" in result
    assert "max_tokens" in result
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 100


def test_param_aliases_empty():
    """Test empty param_aliases dictionary."""
    params = ChatParams(temperature=0.7, max_tokens=100, param_aliases={})

    result = params.to_dict()

    # All parameters should keep original names
    assert "temperature" in result
    assert "max_tokens" in result


def test_param_aliases_with_extra():
    """Test param_aliases combined with extra parameters."""
    params = ChatParams(
        temperature=0.7,
        param_aliases={"temperature": "temp"},
        extra={"custom_param": "value"},
    )

    result = params.to_dict()

    # Aliased parameter
    assert "temp" in result
    assert result["temp"] == 0.7
    assert "temperature" not in result

    # Extra parameter should still be included
    assert "custom_param" in result
    assert result["custom_param"] == "value"


def test_param_aliases_nonexistent_key():
    """Test alias for non-existent parameter key."""
    params = ChatParams(
        temperature=0.7,
        param_aliases={
            "nonexistent_param": "alias"  # This param doesn't exist
        },
    )

    result = params.to_dict()

    # Should not cause errors, just ignore the nonexistent alias
    assert "alias" not in result
    assert "nonexistent_param" not in result
    assert "temperature" in result


def test_param_aliases_exclude_none():
    """Test param_aliases with exclude_none behavior."""
    params = ChatParams(
        temperature=0.7,
        max_tokens=None,  # This is None
        param_aliases={"temperature": "temp", "max_tokens": "max_output_tokens"},
    )

    # With exclude_none=True (default)
    result = params.to_dict(exclude_none=True)

    assert "temp" in result
    assert result["temp"] == 0.7
    assert "max_output_tokens" not in result  # Because max_tokens was None
    assert "temperature" not in result
    assert "max_tokens" not in result


if __name__ == "__main__":
    pytest.main([__file__])
