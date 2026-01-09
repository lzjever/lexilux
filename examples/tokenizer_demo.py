#!/usr/bin/env python
"""
Tokenizer Example

Demonstrates tokenization using Lexilux.
"""

from lexilux import Tokenizer


def main():
    """Main function"""
    # First, let's see what tokenizer files are needed for this model
    print("Listing tokenizer files for the model...")
    try:
        tokenizer_files = Tokenizer.list_tokenizer_files("deepseek-ai/DeepSeek-V3.2")
        print(f"Tokenizer files needed: {len(tokenizer_files)}")
        for file in tokenizer_files:
            print(f"  - {file}")
        print()
    except Exception as e:
        print(f"Could not list files: {e}\n")

    # Initialize tokenizer (offline mode)
    tokenizer = Tokenizer(
        "deepseek-ai/DeepSeek-V3.2", offline=False, cache_dir="~/.cache/lexilux/tokenizer"
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
