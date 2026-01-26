#!/usr/bin/env python
"""
32 Async API - Concurrent Requests

Learn how to use async/await for concurrent API calls.
This is useful for making multiple requests simultaneously.

Level: Advanced Feature
"""

import asyncio
import time

from config_loader import get_chat_config, parse_args

from lexilux import Chat


async def main():
    """Demonstrate async API usage."""
    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    # Example 1: Basic async call
    print("=" * 50)
    print("Example 1: Basic Async Call")
    print("=" * 50)

    chat = Chat(**config)

    print("Calling chat asynchronously...")
    result = await chat.a("Hello, async world!")
    print(f"Response: {result.text}")
    print(f"Tokens: {result.usage.total_tokens}\n")

    # Example 2: Concurrent requests
    print("=" * 50)
    print("Example 2: Concurrent Requests (much faster!)")
    print("=" * 50)

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]

    # Sequential timing (for comparison)
    print("Sequential execution (for comparison):")
    start = time.time()
    for q in questions:
        result = await chat.a(q)
        print(f"  - {result.text[:50]}...")
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s\n")

    # Concurrent execution
    print("Concurrent execution:")
    start = time.time()

    tasks = [chat.a(q) for q in questions]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"  - {result.text[:50]}...")

    concurrent_time = time.time() - start
    print(f"Time: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time / concurrent_time:.1f}x faster!\n")

    # Example 3: Async streaming
    print("=" * 50)
    print("Example 3: Async Streaming")
    print("=" * 50)

    print("Streaming response asynchronously:\n")

    full_response = ""
    async for chunk in chat.astream("Tell me a short joke"):
        if chunk.delta:
            full_response += chunk.delta
            print(chunk.delta, end="", flush=True)
        if chunk.done:
            print(f"\n\nTokens: {chunk.usage.total_tokens}\n")

    # Example 4: Parallel streaming
    print("=" * 50)
    print("Example 4: Parallel Streaming")
    print("=" * 50)

    async def stream_and_collect(prompt: str) -> str:
        """Stream a response and collect the full text."""
        full = ""
        async for chunk in chat.astream(prompt):
            if chunk.delta:
                full += chunk.delta
        return full

    prompts = [
        "Tell me a joke",
        "Give me a quote",
        "Say something wise",
    ]

    print("Streaming multiple requests in parallel:\n")
    start = time.time()

    tasks = [stream_and_collect(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"'{prompt}' → {result[:50]}...")

    print(f"\nTime: {time.time() - start:.2f}s\n")

    # Example 5: Rate limiting with async
    print("=" * 50)
    print("Example 5: Rate Limiting (async semaphore)")
    print("=" * 50)

    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests

    async def limited_chat(prompt: str) -> str:
        """Chat with rate limiting."""
        async with semaphore:
            result = await chat.a(prompt)
            return result.text

    many_prompts = [f"Question {i}" for i in range(5)]

    print(f"Processing {len(many_prompts)} requests with max 2 concurrent:")
    start = time.time()

    tasks = [limited_chat(p) for p in many_prompts]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results, 1):
        print(f"  {i}. {result[:50]}...")

    print(f"\nTime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
