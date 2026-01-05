"""
Chat API client test cases (non-streaming)
"""

import json

import pytest
import responses

from lexilux import Chat, ChatResult
from lexilux.chat.utils import normalize_messages


class TestChatInit:
    """Chat initialization tests"""

    def test_init_with_all_params(self):
        """Test Chat initialization with all parameters"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
            timeout_s=30.0,
            headers={"X-Custom": "value"},
        )
        assert chat.base_url == "https://api.example.com/v1"
        assert chat.api_key == "test-key"
        assert chat.model == "gpt-4"
        assert chat.timeout_s == 30.0
        assert chat.headers["Authorization"] == "Bearer test-key"
        assert chat.headers["X-Custom"] == "value"

    def test_init_without_api_key(self):
        """Test Chat initialization without API key"""
        chat = Chat(base_url="https://api.example.com/v1", model="gpt-4")
        assert chat.api_key is None
        assert "Authorization" not in chat.headers

    def test_init_strips_trailing_slash(self):
        """Test that base_url trailing slash is stripped"""
        chat = Chat(base_url="https://api.example.com/v1/", model="gpt-4")
        assert chat.base_url == "https://api.example.com/v1"


class TestChatNormalizeMessages:
    """Chat message normalization tests"""

    def test_normalize_string(self):
        """Test normalizing a single string"""
        messages = normalize_messages("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_normalize_string_with_system(self):
        """Test normalizing string with system message"""
        messages = normalize_messages("Hello", system="You are helpful")
        assert messages == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_normalize_list_of_strings(self):
        """Test normalizing a list of strings"""
        messages = normalize_messages(["Hello", "World"])
        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]

    def test_normalize_list_of_dicts(self):
        """Test normalizing a list of message dicts"""
        messages = normalize_messages(
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert messages == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_normalize_mixed_list(self):
        """Test normalizing mixed list (strings and dicts)"""
        messages = normalize_messages(["Hello", {"role": "assistant", "content": "Hi!"}])
        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_normalize_invalid_dict(self):
        """Test normalizing invalid message dict"""
        with pytest.raises(ValueError, match="Invalid message dict"):
            normalize_messages([{"invalid": "dict"}])

    def test_normalize_invalid_type(self):
        """Test normalizing invalid message type"""
        with pytest.raises(ValueError, match="Invalid messages type"):
            normalize_messages(123)  # type: ignore


class TestChatCall:
    """Chat __call__ method tests"""

    @responses.activate
    def test_call_with_string(self):
        """Test calling chat with a string"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello! How can I help?"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            status=200,
        )

        result = chat("Hello")
        assert isinstance(result, ChatResult)
        assert result.text == "Hello! How can I help?"
        assert result.usage.total_tokens == 15
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @responses.activate
    def test_call_with_system_message(self):
        """Test calling chat with system message"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "I'm a helpful assistant!"}}],
                "usage": {"total_tokens": 20},
            },
            status=200,
        )

        result = chat("What are you?", system="You are a helpful assistant")
        assert result.text == "I'm a helpful assistant!"

        # Verify request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a helpful assistant"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "What are you?"

    @responses.activate
    def test_call_with_parameters(self):
        """Test calling chat with additional parameters"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        chat(
            "Hello",
            temperature=0.9,
            max_tokens=100,
            stop=".",
            model="gpt-3.5-turbo",
        )

        # Verify request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 100
        assert payload["stop"] == ["."]
        assert payload["model"] == "gpt-3.5-turbo"

    @responses.activate
    def test_call_with_stop_list(self):
        """Test calling chat with stop as a list"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        chat("Hello", stop=[".", "!", "?"])

        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["stop"] == [".", "!", "?"]

    @responses.activate
    def test_call_with_extra_params(self):
        """Test calling chat with extra parameters"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        chat("Hello", extra={"top_p": 0.9, "frequency_penalty": 0.5})

        request = responses.calls[0].request
        payload = json.loads(request.body)
        assert payload["top_p"] == 0.9
        assert payload["frequency_penalty"] == 0.5

    @responses.activate
    def test_call_with_return_raw(self):
        """Test calling chat with return_raw=True"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        response_data = {
            "id": "chat-123",
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"total_tokens": 10},
        }

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json=response_data,
            status=200,
        )

        result = chat("Hello", return_raw=True)
        assert result.raw == response_data

    @responses.activate
    def test_call_without_model(self):
        """Test calling chat without model (should raise error)"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key")

        with pytest.raises(ValueError, match="Model must be specified"):
            chat("Hello")

    @responses.activate
    def test_call_http_error(self):
        """Test handling HTTP errors"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            status=500,
        )

        with pytest.raises(Exception):  # requests.HTTPError
            chat("Hello")

    @responses.activate
    def test_call_no_choices(self):
        """Test handling response with no choices"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={"choices": []},
            status=200,
        )

        with pytest.raises(ValueError, match="No choices in API response"):
            chat("Hello")

    @responses.activate
    def test_call_usage_parsing(self):
        """Test usage parsing from different response formats"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        # Test OpenAI-style usage (prompt_tokens, completion_tokens)
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            status=200,
        )

        result = chat("Hello")
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

        # Test alternative usage format (input_tokens, output_tokens)
        responses.reset()
        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            },
            status=200,
        )

        result = chat("Hello")
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 10
        assert result.usage.total_tokens == 30

    @responses.activate
    def test_call_empty_usage(self):
        """Test handling response with no usage"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Response"}}],
            },
            status=200,
        )

        result = chat("Hello")
        assert result.usage.total_tokens is None
        assert result.usage.input_tokens is None
        assert result.usage.output_tokens is None

    @responses.activate
    def test_call_with_finish_reason_stop(self):
        """Test chat completion with finish_reason='stop'"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        result = chat("Hello")
        assert result.finish_reason == "stop"

    @responses.activate
    def test_call_with_finish_reason_length(self):
        """Test chat completion with finish_reason='length'"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "length"}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        result = chat("Hello")
        assert result.finish_reason == "length"

    @responses.activate
    def test_call_with_finish_reason_content_filter(self):
        """Test chat completion with finish_reason='content_filter'"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        result = chat("Hello")
        assert result.finish_reason == "content_filter"

    @responses.activate
    def test_call_with_finish_reason_none(self):
        """Test chat completion with finish_reason=None (missing field)"""
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

        result = chat("Hello")
        assert result.finish_reason is None

    @responses.activate
    def test_call_with_finish_reason_empty_string(self):
        """Test chat completion with finish_reason='' (defensive handling)"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": ""}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        result = chat("Hello")
        # Empty string should be normalized to None
        assert result.finish_reason is None

    @responses.activate
    def test_call_with_finish_reason_invalid_type(self):
        """Test chat completion with invalid finish_reason type (defensive handling)"""
        chat = Chat(base_url="https://api.example.com/v1", api_key="test-key", model="gpt-4")

        responses.add(
            responses.POST,
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": 123}],
                "usage": {"total_tokens": 10},
            },
            status=200,
        )

        result = chat("Hello")
        # Invalid type should be normalized to None
        assert result.finish_reason is None


class TestChatResult:
    """ChatResult class tests"""

    def test_chat_result_str(self):
        """Test ChatResult string conversion"""
        from lexilux.usage import Usage

        result = ChatResult(text="Hello, world!", usage=Usage(total_tokens=10))
        assert str(result) == "Hello, world!"

    def test_chat_result_repr(self):
        """Test ChatResult representation"""
        from lexilux.usage import Usage

        result = ChatResult(text="Hello", usage=Usage(total_tokens=10))
        repr_str = repr(result)
        assert "ChatResult" in repr_str
        assert "Hello" in repr_str
