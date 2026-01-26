"""Integration tests for reasoning support and wolo adapter example."""

import pytest
from lexilux.chat import Chat
from lexilux.chat.params import ChatParams
from lexilux.chat.models import ChatStreamChunk
from lexilux.usage import Usage


def test_chat_client_reasoning_parameter():
    """Test Chat client accepts include_reasoning parameter."""
    # This tests the API interface, not actual functionality
    # (since we'd need a real API endpoint for full testing)

    chat = Chat(
        base_url="https://api.example.com/v1", api_key="test-key", model="test-model"
    )

    # Test that the stream method accepts include_reasoning parameter
    # We can't actually make the request without a real endpoint,
    # but we can verify the method signature is correct
    import inspect

    stream_signature = inspect.signature(chat.stream)
    assert "include_reasoning" in stream_signature.parameters

    astream_signature = inspect.signature(chat.astream)
    assert "include_reasoning" in astream_signature.parameters

    # Verify default value
    assert stream_signature.parameters["include_reasoning"].default is False
    assert astream_signature.parameters["include_reasoning"].default is False


def test_chat_params_with_reasoning_workflow():
    """Test ChatParams works correctly with the new features."""
    # Test standard usage
    params = ChatParams(temperature=0.8, max_tokens=100)

    result = params.to_dict()
    assert result["temperature"] == 0.8
    assert result["max_tokens"] == 100

    # Test with parameter aliases
    params_with_aliases = ChatParams(
        temperature=0.7, param_aliases={"temperature": "temp"}
    )

    result = params_with_aliases.to_dict()
    assert "temp" in result
    assert result["temp"] == 0.7
    assert "temperature" not in result

    # Test with legitimate extra parameters
    params_with_extra = ChatParams(
        temperature=0.6,
        extra={"response_format": {"type": "json_object"}, "seed": 12345},
    )

    result = params_with_extra.to_dict()
    assert result["temperature"] == 0.6
    assert result["response_format"] == {"type": "json_object"}
    assert result["seed"] == 12345


def test_reasoning_chunk_properties():
    """Test ChatStreamChunk with reasoning properties."""
    # Test chunk with reasoning content
    chunk_with_reasoning = ChatStreamChunk(
        delta="Hello",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        done=False,
        reasoning_content="Let me think about this...",
        reasoning_tokens=20,
    )

    assert chunk_with_reasoning.reasoning_content == "Let me think about this..."
    assert chunk_with_reasoning.reasoning_tokens == 20
    assert chunk_with_reasoning.has_content is True
    assert "reasoning=True" in repr(chunk_with_reasoning)

    # Test chunk without reasoning content
    chunk_normal = ChatStreamChunk(
        delta="Hello",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        done=False,
    )

    assert chunk_normal.reasoning_content is None
    assert chunk_normal.reasoning_tokens is None
    assert chunk_normal.has_content is True
    assert "reasoning=True" not in repr(chunk_normal)


@pytest.mark.skip(
    reason="wolo_adapter_example module has been removed from examples directory"
)
def test_wolo_adapter_example_imports():
    """Test that the wolo adapter example can be imported."""
    # Import the example module
    import sys
    from pathlib import Path

    # Find the examples directory relative to the test file
    test_dir = Path(__file__).parent
    examples_dir = test_dir.parent / "examples"

    # Add examples directory to path if it exists
    if examples_dir.exists():
        sys.path.insert(0, str(examples_dir))

        try:
            from wolo_adapter_example import WoloGLMClient, WoloConfig

            # Test basic instantiation
            config = WoloConfig(
                base_url="https://example.com", api_key="test", model="test-model"
            )

            # Should not raise any errors
            adapter = WoloGLMClient(config)

            # Verify it has the expected methods
            assert hasattr(adapter, "chat_completion")
            assert hasattr(adapter, "_build_opencode_headers")
            assert hasattr(adapter, "_log_request")

            print("✅ Wolo adapter example imports and instantiates correctly")

        except ImportError as e:
            pytest.skip(f"wolo_adapter_example module not found: {e}")
        finally:
            # Clean up path
            if str(examples_dir) in sys.path:
                sys.path.remove(str(examples_dir))
    else:
        pytest.skip(f"Examples directory not found at {examples_dir}")


def test_no_more_error_parameters():
    """Verify that we're not promoting topP/topK error parameters."""
    params = ChatParams()
    result = params.to_dict()

    # These should NOT be in the standard parameters
    assert "topP" not in result
    assert "topK" not in result
    assert "maxOutputTokens" not in result

    # Standard OpenAI parameters should be present
    assert "temperature" in result
    assert "top_p" in result  # This is the CORRECT OpenAI standard

    print("✅ Confirmed: No topP/topK error parameters in standard output")
    print("✅ Standard OpenAI parameters (top_p) present correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
