#!/usr/bin/env python
"""
Rerank Example

Demonstrates document reranking using Lexilux with various scenarios:
- Basic reranking
- Different score formats (positive and negative)
- Sorting rules verification
- Top-k filtering
- Include documents option
"""

from config_loader import get_rerank_config, parse_args

from lexilux import Rerank


def get_rerank_client(config_path=None):
    """Get Rerank client with configuration"""
    args = parse_args()
    config_file = config_path or args.config

    try:
        config = get_rerank_config(config_path=config_file)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading configuration: {e}")
        print("\nUsing default placeholder values. To use real API:")
        print("  1. Create tests/test_endpoints.json with your API credentials")
        print(
            "  2. Or specify a config file: python examples/rerank_demo.py --config /path/to/config.json"
        )
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "rerank-model",
            "mode": "openai",
        }

    return Rerank(**config)


def demo_basic_rerank():
    """Basic reranking example"""
    print("=" * 60)
    print("1. Basic Reranking")
    print("=" * 60)

    rerank = get_rerank_client()

    query = "python http library"
    docs = [
        "urllib is a built-in Python library for HTTP requests",
        "requests is a popular third-party HTTP library for Python",
        "httpx is a modern async HTTP client for Python",
    ]

    result = rerank(query, docs)
    print(f"Query: {query}")
    print("\nRanked results:")
    for i, (idx, score) in enumerate(result.results, 1):
        print(f"  {i}. Doc {idx}: {docs[idx][:50]}... (score: {score:.4f})")

    print(f"\nUsage: {result.usage.total_tokens} tokens")


def demo_top_k():
    """Top-k filtering example"""
    print("\n" + "=" * 60)
    print("2. Top-K Filtering")
    print("=" * 60)

    rerank = get_rerank_client()

    query = "machine learning"
    docs = [
        "Neural networks are computational models",
        "Support vector machines are classification algorithms",
        "Random forests combine multiple decision trees",
        "K-means is a clustering algorithm",
        "Linear regression predicts continuous values",
    ]

    result = rerank(query, docs, top_k=3)
    print(f"Query: {query}")
    print("\nTop 3 results:")
    for i, (idx, score) in enumerate(result.results, 1):
        print(f"  {i}. Doc {idx}: {docs[idx][:50]}... (score: {score:.4f})")


def demo_include_docs():
    """Include documents in results example"""
    print("\n" + "=" * 60)
    print("3. Include Documents in Results")
    print("=" * 60)

    rerank = get_rerank_client()

    query = "data processing"
    docs = [
        "Batch processing handles large datasets",
        "Stream processing handles real-time data",
        "Parallel processing uses multiple cores",
    ]

    result = rerank(query, docs, include_docs=True, top_k=2)
    print(f"Query: {query}")
    print("\nTop 2 results with documents:")
    for i, (idx, score, doc) in enumerate(result.results, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Document: {doc}")


def demo_score_sorting_rules():
    """
    Demonstrate score sorting rules

    Note: Different rerank APIs may return scores in different formats:
    - Positive scores: Higher is better (e.g., 0.95 > 0.80)
    - Negative scores: Less negative is better (e.g., -3.0 > -4.0)

    Lexilux automatically detects the score format and sorts accordingly.
    """
    print("\n" + "=" * 60)
    print("4. Score Sorting Rules")
    print("=" * 60)

    print(
        """
    Lexilux handles different score formats automatically:

    Positive Scores (e.g., 0.95, 0.80, 0.70):
      - Higher score = Better relevance
      - Sorted in descending order: 0.95 > 0.80 > 0.70

    Negative Scores (e.g., -3.0, -4.0, -5.0):
      - Less negative = Better relevance
      - Sorted in descending order: -3.0 > -4.0 > -5.0
      - (Note: -3.0 is mathematically greater than -4.0)

    The library automatically detects which format is used
    and applies the correct sorting order.
    """
    )


def demo_extra_parameters():
    """Example with extra parameters"""
    print("\n" + "=" * 60)
    print("5. Extra Parameters")
    print("=" * 60)

    rerank = get_rerank_client()

    query = "search query"
    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    # Some rerank APIs support additional parameters
    result = rerank(
        query,
        docs,
        top_k=3,
        extra={
            "batch_size": 10,
            "return_documents": False,
        },
    )

    print(f"Query: {query}")
    print("\nTop 3 results:")
    for i, (idx, score) in enumerate(result.results, 1):
        print(f"  {i}. Doc {idx} (score: {score:.4f})")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("Lexilux Rerank Examples")
    print("=" * 60)

    demo_score_sorting_rules()

    # Run demos (will use config if available, otherwise placeholders)
    try:
        demo_basic_rerank()
        demo_top_k()
        demo_include_docs()
        demo_extra_parameters()
    except Exception as e:
        print(
            f"\nNote: Some demos failed (this is expected if using placeholder credentials): {e}"
        )
        print("To run with real APIs:")
        print("  1. Create tests/test_endpoints.json with your API credentials")
        print(
            "  2. Or specify a config file: python examples/rerank_demo.py --config /path/to/config.json"
        )


if __name__ == "__main__":
    main()
