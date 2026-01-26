#!/usr/bin/env python
"""
22 Tokenizer - Count and Analyze Tokens

Learn how to count tokens without making API calls.
This is useful for cost estimation and input length management.

Note: Requires lexilux[tokenizer] extra installation:
    pip install lexilux[tokenizer]

Level: Other APIs
"""

from config_loader import parse_args

from lexilux import Tokenizer


def main():
    """Demonstrate tokenization."""
    parse_args()  # Parse args for consistency (supports --config if needed)

    # Example 1: Basic token counting
    print("=" * 50)
    print("Example 1: Basic Token Counting")
    print("=" * 50)

    # For this example, we'll use offline mode to avoid downloads
    # In real usage, you'd use an actual model name like "Qwen/Qwen2.5-7B-Instruct"
    # For demo purposes, we'll show what would happen

    print("Tokenizing text without API calls...\n")

    # Note: This requires a valid model. For the demo, we'll show the concept.
    try:
        # Try with a common model (will download if not cached)
        tokenizer = Tokenizer("gpt2", offline=False)

        text = "Hello, world!"
        result = tokenizer(text)

        print(f"Text: {text}")
        print(f"Input tokens: {result.usage.input_tokens}")
        print(f"Token IDs: {result.input_ids[0]}\n")

        # Example 2: Comparing texts
        print("=" * 50)
        print("Example 2: Comparing Token Counts")
        print("=" * 50)

        texts = [
            "Hello",
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level programming language.",
        ]

        print("Token counts:")
        for text in texts:
            result = tokenizer(text)
            print(f"  {result.usage.input_tokens:3d} tokens: {text}")

        print("\nNotice: Longer text ≠ proportionally more tokens!")
        print("Words and characters are tokenized differently.\n")

        # Example 3: Batch tokenization
        print("=" * 50)
        print("Example 3: Batch Tokenization")
        print("=" * 50)

        batch_texts = ["Hello", "World", "How are you?"]
        result = tokenizer(batch_texts)

        print(f"Batch tokenized {len(batch_texts)} texts:")
        for i, text in enumerate(batch_texts):
            print(f"  '{text}': {result.usage.input_tokens} tokens")
        print()

        # Example 4: Truncation and padding
        print("=" * 50)
        print("Example 4: Truncation (limiting length)")
        print("=" * 50)

        long_text = "This is a very long text that would be truncated. " * 10
        result = tokenizer(
            long_text,
            max_length=20,
            truncation=True,
        )

        print(f"Original text length: ~{len(long_text.split())} words")
        print(f"Truncated to: {result.usage.input_tokens} tokens")
        print(f"Truncated IDs length: {len(result.input_ids[0])}\n")

        # Example 5: Cost estimation
        print("=" * 50)
        print("Example 5: Cost Estimation")
        print("=" * 50)

        # Estimate tokens for common scenarios
        scenarios = {
            "Short question": "What is the capital of France?",
            "Medium paragraph": (
                "Python is a versatile programming language "
                "that is widely used in web development, "
                "data science, and automation."
            ),
            "Long document": "Chapter 1: Introduction. " * 50,
        }

        # Example pricing (adjust based on your API)
        price_per_1k_tokens = 0.001

        print("Estimated costs (assuming $0.001 per 1K tokens):\n")
        for name, text in scenarios.items():
            result = tokenizer(text)
            tokens = result.usage.input_tokens
            cost = (tokens / 1000) * price_per_1k_tokens
            print(f"{name}:")
            print(f"  Tokens: {tokens}")
            print(f"  Cost: ${cost:.6f}\n")

    except Exception as e:
        print("Note: Tokenization requires transformers library.")
        print(f"Error: {e}")
        print("\nTo install tokenizer support:")
        print("  pip install lexilux[tokenizer]")
        print("\nOr install transformers directly:")
        print("  pip install transformers tokenizers")


if __name__ == "__main__":
    main()
