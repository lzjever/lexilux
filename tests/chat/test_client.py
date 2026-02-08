"""
Chat client streaming cleanup logging tests.
"""

import logging
from unittest.mock import patch, MagicMock

from lexilux import Chat


class TestStreamingCleanupLogging:
    """Tests for streaming cleanup logging."""

    def test_streaming_cleanup_logging_on_early_exit(self):
        """验证流式迭代器提前退出时记录清理日志"""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Mock the _make_streaming_request to avoid actual HTTP calls
        with patch.object(chat, "_make_streaming_request") as mock_request:
            # Create a mock response with iter_lines
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = []
            mock_response.close = MagicMock()
            mock_request.return_value = mock_response

            # Patch logger to verify debug call
            with patch("lexilux.chat.client.logger") as mock_logger:
                iterator = chat.stream("test", max_tokens=10)
                # Consume the iterator to trigger cleanup
                list(iterator)

                # Verify logger.debug was called with cleanup message
                mock_logger.debug.assert_called_once_with(
                    "Closing streaming response and releasing connection"
                )

    def test_streaming_logger_exists(self):
        """验证流式客户端的logger存在"""
        # Verify that the logger is properly defined
        from lexilux.chat import client

        assert hasattr(client, "logger")
        assert isinstance(client.logger, logging.Logger)
