#!/usr/bin/env python
"""
Real API Testing Script

Tests Lexilux with real API endpoints.
Make sure test_endpoints.json is configured with your API credentials.
"""

import json
from pathlib import Path

from lexilux import Chat, Embed, Rerank


def load_test_config():
    """Load test endpoints configuration"""
    # Look for test_endpoints.json in tests directory (or examples directory)
    config_paths = [
        Path(__file__).parent.parent / "tests" / "test_endpoints.json",
        Path(__file__).parent / "test_endpoints.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Test config not found. Tried: {config_paths}\n"
        "Please create test_endpoints.json in tests/ or examples/ directory with your API credentials."
    )


def test_chat(config):
    """Test Chat API"""
    print("=" * 60)
    print("Testing Chat API")
    print("=" * 60)

    chat_config = config["chat"]
    chat = Chat(
        base_url=chat_config["api_base"],
        api_key=chat_config["api_key"],
        model=chat_config["model"],
    )

    # Test 1: Simple call
    print("\n1. Simple chat call:")
    try:
        result = chat("Hello! Please respond with 'Hello from Lexilux!'")
        print(f"   Response: {result.text}")
        print(f"   Usage: {result.usage.total_tokens} tokens")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: With system message
    print("\n2. Chat with system message:")
    try:
        result = chat(
            "What is 2+2?", system="You are a helpful math assistant. Keep answers brief."
        )
        print(f"   Response: {result.text}")
        print(f"   Usage: {result.usage.total_tokens} tokens")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: Streaming
    print("\n3. Streaming chat:")
    try:
        print("   Response: ", end="", flush=True)
        chunk_count = 0
        for chunk in chat.stream("Say 'Hello' in one word", include_usage=True):
            chunk_count += 1
            print(chunk.delta, end="", flush=True)
            if chunk.done:
                usage_tokens = chunk.usage.total_tokens if chunk.usage.total_tokens else "N/A"
                print(f"\n   Usage: {usage_tokens} tokens")
        if chunk_count == 0:
            print("   (No chunks received)")
        print("   ✓ Success")
    except Exception as e:
        import traceback

        print(f"\n   ✗ Error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")


def test_embed(config):
    """Test Embed API"""
    print("\n" + "=" * 60)
    print("Testing Embed API")
    print("=" * 60)

    embed_config = config["embed"]
    embed = Embed(
        base_url=embed_config["api_base"],
        api_key=embed_config["api_key"],
        model=embed_config["model"],
    )

    # Test 1: Single text
    print("\n1. Single text embedding:")
    try:
        result = embed("Hello, world!")
        print(f"   Vector dimensions: {len(result.vectors)}")
        print(f"   First 5 values: {result.vectors[:5]}")
        print(f"   Usage: {result.usage.total_tokens} tokens")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Batch
    print("\n2. Batch embedding:")
    try:
        texts = ["Python is great", "JavaScript is also great"]
        result = embed(texts)
        print(f"   Number of texts: {len(result.vectors)}")
        print(f"   Each vector dimensions: {len(result.vectors[0])}")
        print(f"   Usage: {result.usage.total_tokens} tokens")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")


def test_rerank(config):
    """Test Rerank API"""
    print("\n" + "=" * 60)
    print("Testing Rerank API")
    print("=" * 60)

    rerank_config = config["rerank"]
    rerank = Rerank(
        base_url=rerank_config["api_base"],
        api_key=rerank_config["api_key"],
        model=rerank_config["model"],
    )

    # Test 1: Basic rerank
    print("\n1. Basic rerank:")
    try:
        query = "python http library"
        docs = [
            "urllib is a built-in Python library for HTTP requests",
            "requests is a popular third-party HTTP library for Python",
            "httpx is a modern async HTTP client for Python",
        ]
        result = rerank(query, docs)
        print(f"   Query: {query}")
        print("   Ranked results:")
        for i, (idx, score) in enumerate(result.results, 1):
            doc_text = docs[idx] if idx < len(docs) else f"Doc {idx}"
            print(f"     {i}. Doc {idx}: {doc_text} (score: {score:.4f})")
        print(f"   Usage: {result.usage.total_tokens} tokens")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Note: Rerank endpoint might be different. Try checking the API documentation.")

    # Test 2: With top_k
    print("\n2. Rerank with top_k=2:")
    try:
        query = "python http"
        docs = ["urllib", "requests", "httpx", "aiohttp"]
        result = rerank(query, docs, top_k=2)
        print("   Top 2 results:")
        for i, (idx, score) in enumerate(result.results, 1):
            doc_text = docs[idx] if idx < len(docs) else f"Doc {idx}"
            print(f"     {i}. Doc {idx}: {doc_text} (score: {score:.4f})")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Note: The rerank endpoint might need a different path.")
        print(f"   Current endpoint: {rerank_config['api_base']}/rerank")
        print("   Try checking if it should be /v1/rerank or /rerank/v1")


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("Lexilux Real API Testing")
    print("=" * 60)
    print("\n⚠️  Make sure test_endpoints.json contains valid API credentials!")
    print("⚠️  This file should NOT be committed to git!\n")

    try:
        config = load_test_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Run tests
    try:
        test_chat(config)
    except Exception as e:
        print(f"\nChat test failed: {e}")

    try:
        test_embed(config)
    except Exception as e:
        print(f"\nEmbed test failed: {e}")

    try:
        test_rerank(config)
    except Exception as e:
        print(f"\nRerank test failed: {e}")

    print("\n" + "=" * 60)
    print("Testing Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
