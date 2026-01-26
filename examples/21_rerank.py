#!/usr/bin/env python
"""
21 Document Reranking - Sort by Relevance

Learn how to rerank documents based on their relevance to a query.
This is useful for search results and recommendation systems.

Level: Other APIs
"""

from config_loader import get_chat_config, parse_args

from lexilux import Rerank


def main():
    """Demonstrate document reranking."""
    args = parse_args()

    try:
        config = get_chat_config(config_path=args.config)
        print(
            "Note: Using chat config for rerank. "
            "Configure 'reranker' section for better results."
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "rerank-model",
        }

    rerank = Rerank(**config)

    # Example 1: Basic reranking
    print("=" * 50)
    print("Example 1: Basic Reranking")
    print("=" * 50)

    query = "python programming tutorial"
    documents = [
        "Learn Python from scratch",
        "Python for data science",
        "Introduction to machine learning",
        "Python web development with Flask",
        "JavaScript basics for beginners",
    ]

    result = rerank(query, documents)

    print(f"Query: {query}\n")
    print("Reranked results:")
    for idx, score in result.results:
        print(f"  {idx}. [{score:.2f}] {documents[idx]}")
    print()

    # Example 2: Top-k filtering
    print("=" * 50)
    print("Example 2: Top-K Filtering (get top 3)")
    print("=" * 50)

    result = rerank(query, documents, top_k=3)

    print(f"Query: {query}\n")
    print("Top 3 results:")
    for idx, score in result.results:
        print(f"  {idx}. [{score:.2f}] {documents[idx]}")
    print()

    # Example 3: Include documents in results
    print("=" * 50)
    print("Example 3: Include Documents in Results")
    print("=" * 50)

    result = rerank(query, documents, top_k=3, include_docs=True)

    print(f"Query: {query}\n")
    print("Results with documents:")
    for idx, score, doc in result.results:
        print(f"  [{score:.2f}] {doc}")
    print()

    # Example 4: Search use case
    print("=" * 50)
    print("Example 4: Search Result Reranking")
    print("=" * 50)

    search_query = "how to install python"
    search_results = [
        "Download Python from python.org",
        "Python installation guide for Windows",
        "Best Python IDEs for development",
        "Install Python using package managers",
        "Python vs JavaScript comparison",
    ]

    result = rerank(search_query, search_results, include_docs=True)

    print(f"User searched for: {search_query}\n")
    print("Original order vs Reranked order:\n")

    print("Original:")
    for i, doc in enumerate(search_results):
        print(f"  {i + 1}. {doc}")

    print("\nReranked (most relevant first):")
    for rank, (idx, score, doc) in enumerate(result.results, 1):
        print(f"  {rank}. [{score:.2f}] {doc}")


if __name__ == "__main__":
    main()
