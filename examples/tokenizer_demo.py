#!/usr/bin/env python
"""
Tokenizer Example

Demonstrates tokenization using Lexilux.
"""

from lexilux import Tokenizer


def main():
    """Main function"""
    # Initialize tokenizer (auto-offline mode)
    tokenizer = Tokenizer(
        "Qwen/Qwen2.5-7B-Instruct",
        mode="auto_offline",
    )

    # Single text
    result = tokenizer("Hello, world!")
    print("Text: Hello, world!")
    print(f"Token IDs: {result.input_ids[0]}")
    print(f"Number of tokens: {result.usage.input_tokens}")

    # Batch
    texts = ["Hello", "World", "Python"]
    result = tokenizer(texts)
    print("\nBatch tokenization:")
    for i, (text, ids) in enumerate(zip(texts, result.input_ids)):
        print(f"  {text}: {ids} ({len(ids)} tokens)")

    print(f"\nTotal tokens: {result.usage.total_tokens}")

    # With parameters
    result = tokenizer(
        "This is a long text that might be truncated",
        max_length=10,
        truncation=True,
        padding="max_length",
    )
    print("\nWith truncation and padding:")
    print(f"Token IDs: {result.input_ids[0]}")
    print(f"Length: {len(result.input_ids[0])}")


if __name__ == "__main__":
    main()
