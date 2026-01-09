"""
Comprehensive integration tests for Chat API v2.0.

Tests real-world usage scenarios with explicit history management.
"""

from unittest.mock import Mock, patch

from lexilux import Chat, ChatHistory


class TestChatV2Integration:
    """Integration tests for v2.0 API with explicit history"""

    @patch("lexilux.chat.client.requests.post")
    def test_multi_turn_conversation_with_explicit_history(self, mock_post):
        """Test multi-turn conversation with explicit history management"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # First turn
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [
                {"message": {"content": "Hello! How can I help?"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response1.raise_for_status = Mock()

        # Second turn
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Python is a programming language."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        result1 = chat("Hello", history=history)
        assert result1.text == "Hello! How can I help?"

        # Manually update history for multi-turn conversation (history is immutable)
        history.add_user("Hello")
        history.append_result(result1)

        result2 = chat("What is Python?", history=history)
        assert result2.text == "Python is a programming language."

        # Original history should contain manually added messages
        assert len(history.messages) == 2  # Hello + response
        assert history.messages[0]["content"] == "Hello"
        assert history.messages[1]["content"] == "Hello! How can I help?"

        # Second request should include history (working history is cloned internally)
        second_call_payload = mock_post.call_args_list[1].kwargs["json"]
        messages = second_call_payload["messages"]
        assert len(messages) == 3  # Hello + response + What is Python?
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hello! How can I help?"
        assert messages[2]["content"] == "What is Python?"

    @patch("lexilux.chat.client.requests.post")
    def test_streaming_with_history_accumulation(self, mock_post):
        """Test streaming with history accumulation"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        stream_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " world"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": "!"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(stream_data)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        original_count = len(history.messages)
        iterator = chat.stream("Say hello", history=history)
        list(iterator)

        # Original history should NOT be modified (immutable)
        assert len(history.messages) == original_count

        # Verify result is correct
        result = iterator.result.to_chat_result()
        assert result.text == "Hello world!"

    @patch("lexilux.chat.client.requests.post")
    def test_complete_with_continue(self, mock_post):
        """Test complete() method with automatic continuation"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (truncated)
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Part 1"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response1.raise_for_status = Mock()

        # Continue call (complete)
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": " Part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        original_count = len(history.messages)
        result = chat.complete("Write a story", history=history, max_tokens=50)

        # Should be complete
        assert result.finish_reason == "stop"
        assert "Part 1" in result.text
        assert "Part 2" in result.text

        # Original history should NOT be modified (immutable)
        assert len(history.messages) == original_count

    @patch("lexilux.chat.client.requests.post")
    def test_complete_stream_with_continue(self, mock_post):
        """Test complete_stream() with automatic continuation"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (streaming, truncated)
        stream_data1 = [
            b'data: {"choices": [{"delta": {"content": "Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 1"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "length", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.iter_lines.return_value = iter(stream_data1)
        mock_response1.raise_for_status = Mock()

        # Continue call (streaming, complete)
        stream_data2 = [
            b'data: {"choices": [{"delta": {"content": " Part"}, "index": 0}]}\n',
            b'data: {"choices": [{"delta": {"content": " 2"}, "index": 0}]}\n',
            b'data: {"choices": [{"finish_reason": "stop", "index": 0}]}\n',
            b"data: [DONE]\n",
        ]
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.iter_lines.return_value = iter(stream_data2)
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        iterator = chat.complete_stream("Write a story", history=history, max_tokens=50)
        chunks = list(iterator)

        # Should have chunks from both initial and continue
        assert len(chunks) > 0

        # Result should be complete
        result = iterator.result.to_chat_result()
        assert result.finish_reason == "stop"
        assert "Part 1" in result.text
        assert "Part 2" in result.text

    @patch("lexilux.chat.client.requests.post")
    def test_history_mutable_sequence_operations(self, mock_post):
        """Test ChatHistory as MutableSequence in real usage"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        result1 = chat("Hello", history=history)
        # Manually update history for multi-turn (history is immutable)
        history.add_user("Hello")
        history.append_result(result1)

        result2 = chat("How are you?", history=history)
        # Manually update history
        history.add_user("How are you?")
        history.append_result(result2)

        # Test MutableSequence operations
        assert len(history) == 4

        # Indexing
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

        # Slicing
        first_turn = history[:2]
        assert isinstance(first_turn, ChatHistory)
        assert len(first_turn) == 2

        # Iteration
        messages = list(history)
        assert len(messages) == 4

        # Membership
        assert history[0] in history

        # Modification
        history[0] = {"role": "user", "content": "Modified"}
        assert history[0]["content"] == "Modified"

        # Deletion
        del history[0]
        assert len(history) == 3

    @patch("lexilux.chat.client.requests.post")
    def test_history_clone_and_merge(self, mock_post):
        """Test history cloning and merging"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history1 = ChatHistory()
        result1 = chat("Hello", history=history1)
        # Manually update history (immutable)
        history1.add_user("Hello")
        history1.append_result(result1)

        # Clone history
        history2 = history1.clone()
        assert history2 is not history1
        assert len(history2) == len(history1)
        assert history2[0]["content"] == history1[0]["content"]

        # Modify clone (should not affect original)
        history2[0] = {"role": "user", "content": "Modified"}
        assert history1[0]["content"] == "Hello"  # Original unchanged

        # Merge histories
        history3 = ChatHistory()
        result2 = chat("How are you?", history=history3)
        # Manually update history
        history3.add_user("How are you?")
        history3.append_result(result2)

        combined = history1 + history3
        assert isinstance(combined, ChatHistory)
        assert len(combined) == 4  # 2 from history1 + 2 from history3

    @patch("lexilux.chat.client.requests.post")
    def test_continue_with_explicit_history(self, mock_post):
        """Test continue functionality with explicit history"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        # Initial call (truncated)
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [{"message": {"content": "Part 1"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        mock_response1.raise_for_status = Mock()

        # Continue call
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": " Part 2"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response2.raise_for_status = Mock()

        mock_post.side_effect = [mock_response1, mock_response2]

        history = ChatHistory()
        result1 = chat("Write a story", history=history, max_tokens=50)
        assert result1.finish_reason == "length"

        # Continue using ChatContinue (continue_if_needed removed)
        from lexilux.chat.continue_ import ChatContinue

        full_result = ChatContinue.continue_request(chat, result1, history=history)
        assert full_result.finish_reason == "stop"
        assert "Part 1" in full_result.text
        assert "Part 2" in full_result.text

    @patch("lexilux.chat.client.requests.post")
    def test_multiple_independent_histories(self, mock_post):
        """Test using multiple independent history objects"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Two independent conversations
        history1 = ChatHistory()
        history2 = ChatHistory()

        result1 = chat("Hello", history=history1)
        result2 = chat("Hi", history=history2)

        # Manually update histories (immutable)
        history1.add_user("Hello")
        history1.append_result(result1)
        history2.add_user("Hi")
        history2.append_result(result2)

        # Histories should be independent
        assert len(history1) == 2
        assert len(history2) == 2
        assert history1[0]["content"] == "Hello"
        assert history2[0]["content"] == "Hi"

        # Continue conversation in history1
        result3 = chat("How are you?", history=history1)
        history1.add_user("How are you?")
        history1.append_result(result3)
        assert len(history1) == 4
        assert len(history2) == 2  # history2 unchanged

    @patch("lexilux.chat.client.requests.post")
    def test_history_query_methods(self, mock_post):
        """Test history query methods in real usage"""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        history = ChatHistory()
        result1 = chat("Hello", history=history)
        # Manually update history (immutable)
        history.add_user("Hello")
        history.append_result(result1)

        result2 = chat("How are you?", history=history)
        # Manually update history
        history.add_user("How are you?")
        history.append_result(result2)

        # Query methods
        user_messages = history.get_user_messages()
        assert user_messages == ["Hello", "How are you?"]

        assistant_messages = history.get_assistant_messages()
        assert len(assistant_messages) == 2
        assert assistant_messages[0] == "Response"
        assert assistant_messages[1] == "Response"

        last_message = history.get_last_message()
        assert last_message["role"] == "assistant"

        last_user_message = history.get_last_user_message()
        assert last_user_message == "How are you?"
