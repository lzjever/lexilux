"""
Tests for example scripts.

This module tests all example scripts in the examples/ directory.
Tests use mocked API responses to avoid requiring actual API keys.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import responses

# Add examples directory to path so we can import the modules
examples_dir = Path(__file__).parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


# ============================================================================
# Helper function to import modules with numeric names
# ============================================================================


def import_example_module(name: str):
    """Import an example module by name, handling numeric prefixes."""
    return importlib.import_module(f"examples.{name}")


# ============================================================================
# Shared fixtures and helpers
# ============================================================================


@pytest.fixture
def mock_config():
    """Mock configuration for examples."""
    return {
        "base_url": "https://api.example.com/v1",
        "api_key": "test-api-key",
        "model": "gpt-4",
    }


@pytest.fixture
def mock_chat_response():
    """Standard mock chat API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! This is a mock response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_stream_chunks():
    """Mock streaming response chunks."""
    return [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world!"}, "finish_reason":"stop"}]}',
        "data: [DONE]",
    ]


def mock_parse_args():
    """Mock parse_args to return default config path."""

    class Args:
        config = None

    return Args()


# ============================================================================
# Syntax import tests - verify all examples can be imported
# ============================================================================


@pytest.mark.example
class TestExampleImports:
    """Test that all example modules can be imported without errors."""

    def test_import_01_hello_world(self):
        """Test 01_hello_world.py can be imported."""
        import_example_module("01_hello_world")

    def test_import_02_system_message(self):
        """Test 02_system_message.py can be imported."""
        import_example_module("02_system_message")

    def test_import_10_streaming(self):
        """Test 10_streaming.py can be imported."""
        import_example_module("10_streaming")

    def test_import_11_conversation(self):
        """Test 11_conversation.py can be imported."""
        import_example_module("11_conversation")

    def test_import_12_chat_params(self):
        """Test 12_chat_params.py can be imported."""
        import_example_module("12_chat_params")

    def test_import_20_embedding(self):
        """Test 20_embedding.py can be imported."""
        import_example_module("20_embedding")

    def test_import_21_rerank(self):
        """Test 21_rerank.py can be imported."""
        import_example_module("21_rerank")

    def test_import_22_tokenizer(self):
        """Test 22_tokenizer.py can be imported."""
        try:
            import transformers  # noqa: F401

            import_example_module("22_tokenizer")
        except ImportError:
            # Skip if transformers is not installed
            pytest.skip("transformers not installed")

    def test_import_30_function_calling(self):
        """Test 30_function_calling.py can be imported."""
        import_example_module("30_function_calling")

    def test_import_31_multimodal(self):
        """Test 31_multimodal.py can be imported."""
        import_example_module("31_multimodal")

    def test_import_32_async(self):
        """Test 32_async.py can be imported."""
        import_example_module("32_async")

    def test_import_40_chat_history(self):
        """Test 40_chat_history.py can be imported."""
        import_example_module("40_chat_history")

    def test_import_41_auto_continue(self):
        """Test 41_auto_continue.py can be imported."""
        import_example_module("41_auto_continue")

    def test_import_42_error_handling(self):
        """Test 42_error_handling.py can be imported."""
        import_example_module("42_error_handling")

    def test_import_43_custom_formatting(self):
        """Test 43_custom_formatting.py can be imported."""
        import_example_module("43_custom_formatting")

    def test_import_config_loader(self):
        """Test config_loader.py can be imported."""
        import_example_module("config_loader")


# ============================================================================
# Example execution tests - verify examples can run with mocked APIs
# ============================================================================


@pytest.mark.example
class TestExampleExecution:
    """Test that example scripts can execute with mocked API responses.

    Note: Only basic examples are tested here. Some examples may use
    outdated or experimental APIs. The import tests verify syntax
    correctness for all examples.
    """

    @responses.activate
    def test_01_hello_world_execution(self, mock_config, mock_chat_response):
        """Test 01_hello_world.py runs correctly with mocked API."""
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=mock_chat_response,
            status=200,
        )

        example = import_example_module("01_hello_world")

        # Patch the module's attributes directly after import
        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                example.main()

    @responses.activate
    def test_02_system_message_execution(self, mock_config, mock_chat_response):
        """Test 02_system_message.py runs correctly with mocked API."""
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=mock_chat_response,
            status=200,
        )

        example = import_example_module("02_system_message")

        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                example.main()

    @responses.activate
    def test_10_streaming_execution(self, mock_config):
        """Test 10_streaming.py runs correctly with mocked API."""
        chunks = [
            'data: {"choices":[{"delta":{"content":"Joke:"}}]}',
            'data: {"choices":[{"delta":{"content":" Why did the programmer quit?"}}]}',
            'data: {"choices":[{"delta":{"content":" He didn\'t get arrays."}, "finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        # Use responses mock with body instead of stream parameter
        # Build the streaming body with newlines between chunks
        body = "\n".join(chunks)

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            body=body,
            status=200,
            content_type="text/event-stream",
        )

        # Add responses for additional streaming calls
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            body=body,
            status=200,
            content_type="text/event-stream",
        )

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            body=body,
            status=200,
            content_type="text/event-stream",
        )

        example = import_example_module("10_streaming")

        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                example.main()

    @responses.activate
    def test_11_conversation_execution(self, mock_config, mock_chat_response):
        """Test 11_conversation.py runs correctly with mocked API."""
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=mock_chat_response,
            status=200,
        )

        example = import_example_module("11_conversation")

        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                # Mock input to exit immediately from interactive loop
                with patch("builtins.input", return_value="quit"):
                    example.main()

    @responses.activate
    def test_12_chat_params_execution(self, mock_config, mock_chat_response):
        """Test 12_chat_params.py runs correctly with mocked API."""
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=mock_chat_response,
            status=200,
        )

        example = import_example_module("12_chat_params")

        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                example.main()

    @responses.activate
    def test_20_embedding_execution(self, mock_config):
        """Test 20_embedding.py runs correctly with mocked API."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}],
            "usage": {"total_tokens": 10},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=mock_response,
            status=200,
        )

        mock_batch_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
                {"embedding": [0.7, 0.8, 0.9]},
                {"embedding": [1.0, 1.1, 1.2]},
            ],
            "usage": {"total_tokens": 40},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=mock_batch_response,
            status=200,
        )

        responses.add(
            responses.POST,
            "https://api.example.com/v1/embeddings",
            json=mock_batch_response,
            status=200,
        )

        example = import_example_module("20_embedding")

        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                example.main()

    @responses.activate
    def test_30_function_calling_execution(self, mock_config):
        """Test 30_function_calling.py runs correctly with mocked API."""
        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris", "unit": "celsius"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 30},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=tool_call_response,
            status=200,
        )

        final_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The weather in Paris is 22°C.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 50},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=final_response,
            status=200,
        )

        parallel_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 40},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=parallel_response,
            status=200,
        )

        example = import_example_module("30_function_calling")

        with patch.object(example, "parse_args", return_value=mock_parse_args()):
            with patch.object(example, "get_chat_config", return_value=mock_config):
                example.main()
