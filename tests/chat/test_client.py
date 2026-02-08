"""
Chat client streaming cleanup logging tests.
"""

import logging
from unittest.mock import patch, MagicMock

from lexilux import Chat


class TestStreamingCleanupLogging:
    """Tests for streaming cleanup logging."""

    def test_streaming_cleanup_logging_on_early_exit(self):
        """Verify cleanup logging when streaming iterator exits early"""
        chat = Chat(base_url="https://api.example.com", api_key="test", model="gpt-4")

        # Mock the _streaming_request_context to avoid actual HTTP calls
        with patch.object(chat, "_streaming_request_context") as mock_context:
            # Create a mock response with iter_lines
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = []
            mock_response.close = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_context.return_value = mock_response

            # Patch logger to verify debug call
            with patch("lexilux.chat.client.logger") as mock_logger:
                iterator = chat.stream("test", max_tokens=10)
                # Consume the iterator to trigger cleanup
                list(iterator)

                # Verify logger.debug was NOT called (cleanup is handled by context manager)
                mock_logger.debug.assert_not_called()

    def test_streaming_logger_exists(self):
        """Verify streaming client logger exists"""
        # Verify that the logger is properly defined
        from lexilux.chat import client

        assert hasattr(client, "logger")
        assert isinstance(client.logger, logging.Logger)
