#!/usr/bin/env python
"""
Chat Parameters Demo

Demonstrates various parameter configurations and their effects on chat completions.
Shows how different parameters (max_tokens, stop sequences, temperature, etc.)
affect the output and finish_reason.
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat, ChatParams


def get_chat_client(config_path=None):
    """Get Chat client with configuration"""
    args = parse_args()
    config_file = config_path or args.config

    try:
        config = get_chat_config(config_path=config_file)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading configuration: {e}")
        print("\nUsing default placeholder values. To use real API:")
        print("  1. Create tests/test_endpoints.json with your API credentials")
        print(
            "  2. Or specify a config file: python examples/chat_params_demo.py --config /path/to/config.json"
        )
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    return Chat(**config)


def demo_max_tokens():
    """Demonstrate max_tokens parameter and finish_reason"""
    print("=" * 70)
    print("Demo 1: max_tokens parameter")
    print("=" * 70)

    chat = get_chat_client()

    # Request with small max_tokens to trigger length limit
    print("\n1. Request with max_tokens=10 (should trigger 'length' finish_reason):")
    result = chat(
        "Write a detailed explanation of how neural networks work",
        max_tokens=10,
    )
    print(f"   Output: {result.text[:100]}...")
    print(f"   Finish reason: {result.finish_reason}")
    print(f"   Tokens used: {result.usage.total_tokens}")

    # Request with larger max_tokens
    print("\n2. Request with max_tokens=100 (should complete normally):")
    result = chat(
        "Write a short explanation of neural networks",
        max_tokens=100,
    )
    print(f"   Output: {result.text[:100]}...")
    print(f"   Finish reason: {result.finish_reason}")
    print(f"   Tokens used: {result.usage.total_tokens}")


def demo_stop_sequences():
    """Demonstrate stop sequences parameter"""
    print("\n" + "=" * 70)
    print("Demo 2: stop sequences parameter")
    print("=" * 70)

    chat = get_chat_client()

    # Request with stop sequence
    print("\n1. Request with stop='.' (should stop at first period):")
    result = chat(
        "Count from 1 to 10. Then count backwards.",
        stop=".",
    )
    print(f"   Output: {result.text}")
    print(f"   Finish reason: {result.finish_reason}")
    print("   Note: Output should not contain the stop sequence '.'")

    # Request with multiple stop sequences
    print("\n2. Request with stop=['3', '4'] (should stop at first occurrence):")
    result = chat(
        "Count from 1 to 5: 1, 2, 3, 4, 5",
        stop=["3", "4"],
    )
    print(f"   Output: {result.text}")
    print(f"   Finish reason: {result.finish_reason}")


def demo_temperature():
    """Demonstrate temperature parameter effects"""
    print("\n" + "=" * 70)
    print("Demo 3: temperature parameter (creativity control)")
    print("=" * 70)

    chat = get_chat_client()

    prompt = "Write a creative story about a robot learning to paint."

    # Low temperature (more deterministic)
    print("\n1. Low temperature (0.2) - more focused and deterministic:")
    result_low = chat(prompt, temperature=0.2, max_tokens=50)
    print(f"   Output: {result_low.text[:150]}...")
    print(f"   Finish reason: {result_low.finish_reason}")

    # High temperature (more random)
    print("\n2. High temperature (1.5) - more creative and random:")
    result_high = chat(prompt, temperature=1.5, max_tokens=50)
    print(f"   Output: {result_high.text[:150]}...")
    print(f"   Finish reason: {result_high.finish_reason}")

    print("\n   Note: Same prompt, different outputs due to temperature!")


def demo_chatparams():
    """Demonstrate ChatParams dataclass usage"""
    print("\n" + "=" * 70)
    print("Demo 4: ChatParams dataclass for structured configuration")
    print("=" * 70)

    chat = get_chat_client()

    # Create parameter configuration
    params = ChatParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        presence_penalty=0.3,
        frequency_penalty=0.2,
    )

    print("\n1. Using ChatParams for structured configuration:")
    result = chat("Tell me about artificial intelligence", params=params)
    print(f"   Output: {result.text[:150]}...")
    print(f"   Finish reason: {result.finish_reason}")
    print(f"   Tokens: {result.usage.total_tokens}")

    # Override params with direct arguments
    print("\n2. Overriding ChatParams with direct arguments:")
    result = chat(
        "Tell me about artificial intelligence",
        params=params,
        temperature=0.5,  # Override params.temperature
    )
    print(f"   Output: {result.text[:150]}...")
    print(f"   Finish reason: {result.finish_reason}")


def demo_penalties():
    """Demonstrate presence_penalty and frequency_penalty"""
    print("\n" + "=" * 70)
    print("Demo 5: Penalties (presence_penalty and frequency_penalty)")
    print("=" * 70)

    chat = get_chat_client()

    prompt = "Write a paragraph about Python programming."

    # Without penalties
    print("\n1. Without penalties (default):")
    result_default = chat(prompt, max_tokens=50)
    print(f"   Output: {result_default.text[:150]}...")
    print(f"   Finish reason: {result_default.finish_reason}")

    # With presence_penalty (encourages new topics)
    print("\n2. With presence_penalty=0.6 (encourages new topics):")
    result_presence = chat(prompt, presence_penalty=0.6, max_tokens=50)
    print(f"   Output: {result_presence.text[:150]}...")
    print(f"   Finish reason: {result_presence.finish_reason}")

    # With frequency_penalty (reduces repetition)
    print("\n3. With frequency_penalty=0.6 (reduces repetition):")
    result_frequency = chat(prompt, frequency_penalty=0.6, max_tokens=50)
    print(f"   Output: {result_frequency.text[:150]}...")
    print(f"   Finish reason: {result_frequency.finish_reason}")


def demo_streaming_with_params():
    """Demonstrate streaming with parameters"""
    print("\n" + "=" * 70)
    print("Demo 6: Streaming with parameters")
    print("=" * 70)

    chat = get_chat_client()

    # Streaming with max_tokens
    print("\n1. Streaming with max_tokens=20:")
    print("   Output: ", end="", flush=True)
    chunks = []
    for chunk in chat.stream("Count from 1 to 10", max_tokens=20):
        print(chunk.delta, end="", flush=True)
        chunks.append(chunk)

    # Find the chunk with done=True to get finish_reason
    # Priority: chunk with finish_reason > last done chunk > last chunk
    done_chunks = [chunk for chunk in chunks if chunk.done]
    if done_chunks:
        # Prefer chunk with finish_reason (usually the one before [DONE])
        final_chunk = next(
            (chunk for chunk in done_chunks if chunk.finish_reason is not None),
            done_chunks[-1],  # Fallback to last done chunk if no finish_reason
        )
        print(f"\n   Finish reason: {final_chunk.finish_reason}")
        print(f"   Tokens: {final_chunk.usage.total_tokens}")
    else:
        # Fallback to last chunk if no done chunk found
        final_chunk = chunks[-1]
        print(
            f"\n   Finish reason: {final_chunk.finish_reason} (done={final_chunk.done})"
        )
        print(f"   Tokens: {final_chunk.usage.total_tokens}")

    # Streaming with stop sequence
    print("\n2. Streaming with stop='5':")
    print("   Output: ", end="", flush=True)
    chunks = []
    for chunk in chat.stream("Count from 1 to 10", stop="5"):
        print(chunk.delta, end="", flush=True)
        chunks.append(chunk)

    # Find the chunk with done=True to get finish_reason
    # Priority: chunk with finish_reason > last done chunk > last chunk
    done_chunks = [chunk for chunk in chunks if chunk.done]
    if done_chunks:
        # Prefer chunk with finish_reason (usually the one before [DONE])
        final_chunk = next(
            (chunk for chunk in done_chunks if chunk.finish_reason is not None),
            done_chunks[-1],  # Fallback to last done chunk if no finish_reason
        )
        print(f"\n   Finish reason: {final_chunk.finish_reason}")
    else:
        # Fallback to last chunk if no done chunk found
        final_chunk = chunks[-1]
        print(
            f"\n   Finish reason: {final_chunk.finish_reason} (done={final_chunk.done})"
        )


def demo_comparison():
    """Compare different parameter combinations"""
    print("\n" + "=" * 70)
    print("Demo 7: Comparing different parameter combinations")
    print("=" * 70)

    chat = get_chat_client()

    prompt = "Write a haiku about programming."

    # Configuration 1: Default
    print("\n1. Default parameters:")
    result1 = chat(prompt, max_tokens=30)
    print(f"   Output: {result1.text}")
    print(f"   Finish reason: {result1.finish_reason}")

    # Configuration 2: Low temperature, no penalties
    print("\n2. Low temperature (0.2), no penalties:")
    result2 = chat(prompt, temperature=0.2, max_tokens=30)
    print(f"   Output: {result2.text}")
    print(f"   Finish reason: {result2.finish_reason}")

    # Configuration 3: High temperature, with penalties
    print("\n3. High temperature (1.2), with penalties:")
    result3 = chat(
        prompt,
        temperature=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.3,
        max_tokens=30,
    )
    print(f"   Output: {result3.text}")
    print(f"   Finish reason: {result3.finish_reason}")

    print("\n   Note: Same prompt, different outputs due to parameter differences!")


def main():
    """Main function to run all demos"""
    print("\n" + "=" * 70)
    print("Chat Parameters Demo")
    print("=" * 70)
    print("\nThis demo shows how different parameters affect chat completions.")
    print("Note: Replace 'your-api-key' with your actual API key to run.")
    print("\n" + "=" * 70)

    try:
        demo_max_tokens()
        demo_stop_sequences()
        demo_temperature()
        demo_chatparams()
        demo_penalties()
        demo_streaming_with_params()
        demo_comparison()

        print("\n" + "=" * 70)
        print("All demos completed!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: This demo requires a valid API key and endpoint.")
        print("Please update the API credentials in the code.")


if __name__ == "__main__":
    main()
