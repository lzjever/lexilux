"""Concurrent safety tests for lexilux clients."""

import asyncio
import threading
import pytest
from unittest.mock import Mock, patch
from lexilux import Chat


class TestConcurrentSync:
    """Test thread safety of sync clients."""

    def test_concurrent_sync_requests_same_client(self):
        """Test multiple threads using same Chat instance."""
        # Mock to avoid actual API calls
        with patch("lexilux._base.BaseAPIClient._make_request") as mock_req:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.ok = True
            mock_response.json.return_value = {
                "id": "test",
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }
            mock_req.return_value = mock_response

            chat = Chat(
                base_url="https://api.example.com", api_key="test", model="gpt-4"
            )

            results = []
            errors = []

            def make_request(n):
                try:
                    result = chat(f"Hello {n}")
                    results.append(result.text)
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(20):
                t = threading.Thread(target=make_request, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 20

    def test_concurrent_sync_streaming_requests_same_client(self):
        """Test multiple threads using same Chat instance for streaming."""
        with patch(
            "lexilux._base.BaseAPIClient._streaming_request_context"
        ) as mock_ctx:
            # Create a mock response that yields SSE data
            class MockStreamResponse:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def iter_lines(self):
                    # Yield SSE-like data
                    yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
                    yield b'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}\n\n'
                    yield b"data: [DONE]\n\n"

            mock_ctx.return_value = MockStreamResponse()

            chat = Chat(
                base_url="https://api.example.com", api_key="test", model="gpt-4"
            )

            results = []
            errors = []

            def make_stream_request(n):
                try:
                    chunks = []
                    for chunk in chat.stream(f"Hello {n}"):
                        chunks.append(chunk)
                    results.append(len(chunks))
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(10):
                t = threading.Thread(target=make_stream_request, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 10


class TestConcurrentAsync:
    """Test asyncio concurrency of async clients."""

    @pytest.mark.asyncio
    async def test_concurrent_async_requests_same_client(self):
        """Test multiple concurrent async requests."""
        with patch("lexilux._base.BaseAPIClient._ado_request") as mock_req:

            async def mock_async_request(*args, **kwargs):
                # Simulate async delay
                await asyncio.sleep(0.01)
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.is_success = True
                mock_response.json.return_value = {
                    "id": "test",
                    "choices": [
                        {"message": {"content": "Hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                }
                return mock_response

            mock_req.side_effect = mock_async_request

            chat = Chat(
                base_url="https://api.example.com", api_key="test", model="gpt-4"
            )

            async def make_request(n):
                result = await chat.acall(f"Hello {n}")
                return result.text

            results = await asyncio.gather(*[make_request(i) for i in range(20)])
            assert len(results) == 20
            assert all(r == "Hello" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self):
        """Test multiple concurrent streaming requests."""

        # Create a factory function that returns a fresh async generator for each call
        def mock_stream_gen_factory():
            async def mock_stream_gen():
                chunks = [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
                    'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}\n\n',
                    "data: [DONE]\n\n",
                ]
                for chunk in chunks:
                    await asyncio.sleep(0.01)
                    yield chunk

            return mock_stream_gen()

        with patch(
            "lexilux._base.BaseAPIClient._amake_streaming_request",
            side_effect=lambda *args, **kwargs: mock_stream_gen_factory(),
        ):
            chat = Chat(
                base_url="https://api.example.com", api_key="test", model="gpt-4"
            )

            async def stream_request(n):
                iterator = await chat.astream(f"Hello {n}")
                count = 0
                async for _ in iterator:
                    count += 1
                    if count >= 2:
                        break
                return count

            results = await asyncio.gather(*[stream_request(i) for i in range(10)])
            assert all(r >= 1 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_requests(self):
        """Test concurrent mix of regular and streaming async requests."""

        # Create a factory function that returns a fresh async generator for each call
        def mock_stream_gen_factory():
            async def mock_stream_gen():
                yield 'data: {"choices": [{"delta": {"content": "Stream"}}]}\n\n'
                yield 'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}\n\n'
                yield "data: [DONE]\n\n"

            return mock_stream_gen()

        with (
            patch("lexilux._base.BaseAPIClient._ado_request") as mock_req,
            patch(
                "lexilux._base.BaseAPIClient._amake_streaming_request",
                side_effect=lambda *args, **kwargs: mock_stream_gen_factory(),
            ),
        ):

            async def mock_async_request(*args, **kwargs):
                await asyncio.sleep(0.01)
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.is_success = True
                mock_response.json.return_value = {
                    "id": "test",
                    "choices": [
                        {"message": {"content": "Response"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                }
                return mock_response

            mock_req.side_effect = mock_async_request

            chat = Chat(
                base_url="https://api.example.com", api_key="test", model="gpt-4"
            )

            async def regular_request(n):
                result = await chat.acall(f"Request {n}")
                return result.text

            async def stream_request(n):
                iterator = await chat.astream(f"Stream {n}")
                chunks = []
                async for chunk in iterator:
                    chunks.append(chunk)
                    if len(chunks) >= 1:
                        break
                return len(chunks)

            # Mix of regular and streaming requests
            tasks = []
            for i in range(10):
                if i % 2 == 0:
                    tasks.append(regular_request(i))
                else:
                    tasks.append(stream_request(i))

            results = await asyncio.gather(*tasks)
            assert len(results) == 10

    @pytest.mark.asyncio
    async def test_async_client_multiple_acreate_calls(self):
        """Test that multiple async clients can be created and used concurrently."""

        async def create_and_use_client(n):
            chat = Chat(
                base_url="https://api.example.com", api_key="test", model="gpt-4"
            )

            with patch.object(chat, "_ado_request") as mock_req:

                async def mock_async_request(*args, **kwargs):
                    await asyncio.sleep(0.01)
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.is_success = True
                    mock_response.json.return_value = {
                        "id": f"test-{n}",
                        "choices": [
                            {
                                "message": {"content": f"Response {n}"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                    }
                    return mock_response

                mock_req.side_effect = mock_async_request

                result = await chat.acall(f"Hello {n}")
                return result.text

        results = await asyncio.gather(*[create_and_use_client(i) for i in range(10)])
        assert len(results) == 10
        assert all(f"Response {i}" in r for i, r in enumerate(results))
