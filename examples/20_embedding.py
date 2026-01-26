#!/usr/bin/env python
"""
20 Text Embedding - Convert Text to Numbers

Learn how to convert text into vector embeddings.
Embeddings are useful for semantic search, similarity, and more.

Level: Other APIs
"""

from config_loader import get_chat_config, parse_args

from lexilux import Embed


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    import math

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (magnitude_a * magnitude_b)


def main():
    """Demonstrate text embedding."""
    args = parse_args()

    # For embedding, we need to use the embed config
    try:
        config = get_chat_config(config_path=args.config)
        print(
            "Note: Using chat config for embedding. "
            "Configure 'embedding' section for better results."
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "text-embedding-ada-002",
        }

    embed = Embed(**config)

    # Example 1: Single text embedding
    print("=" * 50)
    print("Example 1: Single Text Embedding")
    print("=" * 50)

    text = "Hello, world!"
    result = embed(text)

    print(f"Text: {text}")
    print(f"Vector dimension: {len(result.vectors)}")
    print(f"First 5 values: {result.vectors[:5]}")
    print(f"Tokens used: {result.usage.total_tokens}\n")

    # Example 2: Batch embeddings
    print("=" * 50)
    print("Example 2: Batch Embeddings")
    print("=" * 50)

    texts = [
        "The cat sits on the mat",
        "The dog plays in the park",
        "A feline is resting on a rug",
        "I love programming in Python",
    ]

    result = embed(texts)

    print(f"Embedded {len(texts)} texts")
    print(f"Each vector has {len(result.vectors[0])} dimensions")
    print(f"Total tokens used: {result.usage.total_tokens}\n")

    # Example 3: Semantic similarity
    print("=" * 50)
    print("Example 3: Semantic Similarity")
    print("=" * 50)

    # Get embeddings
    result = embed(texts)
    embeddings = result.vectors

    # Calculate similarities
    print("Similarity matrix:")
    print("-" * 50)
    for i, text_i in enumerate(texts):
        for j, text_j in enumerate(texts):
            if i <= j:
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                print(f"{similarity:.2f}".rjust(6), end=" ")
            else:
                print("     ".rjust(6), end=" ")
        print()

    print("\nText labels:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text[:40]}...")

    print("\nHigher values = more similar meaning!")
    print("Notice how 'cat' and 'feline' are most similar.\n")


if __name__ == "__main__":
    main()
