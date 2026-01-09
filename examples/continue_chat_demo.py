#!/usr/bin/env python
"""
Comprehensive Continue Chat Demo

This example demonstrates various continue chat scenarios that users might encounter
in production environments. It intentionally uses small max_tokens to trigger length
interruptions and tests different continuation strategies.

Scenarios covered:
1. Basic continue with single interruption
2. Multiple continues (chain of interruptions)
3. Streaming continue
4. Manual continue without auto_merge
5. Continue with custom prompts
6. Large input + small output limit (common in production)
7. Continue after concatenating output back to input
8. Error handling and edge cases
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import get_chat_config, parse_args

from lexilux import Chat, ChatHistory, ChatContinue


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result_info(result, label: str = "Result"):
    """Print information about a result."""
    print(f"{label}:")
    print(f"  Text length: {len(result.text)} characters")
    print(f"  Finish reason: {result.finish_reason}")
    print(
        f"  Usage: {result.usage.total_tokens} tokens (input: {result.usage.input_tokens}, output: {result.usage.output_tokens})"
    )
    if result.finish_reason == "length":
        print(f"  ⚠️  TRUNCATED - needs continuation")
    print()


def scenario_1_basic_continue(chat: Chat):
    """Scenario 1: Basic continue with single interruption."""
    print_section("Scenario 1: Basic Continue (Single Interruption)")

    history = ChatHistory()

    # Request a long story with very small max_tokens to force interruption
    print("Requesting a long story with max_tokens=50 (will likely be truncated)...")
    result = chat(
        "Write a detailed story about a space explorer discovering a new planet. Make it at least 500 words.",
        history=history,
        max_tokens=50,  # Very small to force truncation
    )

    print_result_info(result, "Initial Result")
    print(f"First 100 chars: {result.text[:100]}...")

    if result.finish_reason == "length":
        print("\n🔄 Continuing generation...")
        # Use ChatContinue.continue_request for basic continue
        full_result = ChatContinue.continue_request(
            chat,
            result,
            history=history,
            max_continues=1,
        )

        print_result_info(full_result, "After Continue")
        print(f"Full text length: {len(full_result.text)} characters")
        print(f"First 200 chars: {full_result.text[:200]}...")

        if full_result.finish_reason == "length":
            print("⚠️  Still truncated after 1 continue")
        else:
            print("✅ Successfully completed after continue")
    else:
        print("ℹ️  No continuation needed (finish_reason != 'length')")


def scenario_2_multiple_continues(chat: Chat):
    """Scenario 2: Multiple continues (chain of interruptions)."""
    print_section("Scenario 2: Multiple Continues (Chain of Interruptions)")

    history = ChatHistory()

    # Request something very long with tiny max_tokens to force multiple continues
    print("Requesting a very long response with max_tokens=30 (will need multiple continues)...")
    result = chat(
        "Write a comprehensive guide on Python programming covering: variables, functions, classes, modules, and best practices. Make it detailed and at least 1000 words.",
        history=history,
        max_tokens=30,  # Very small to force multiple continues
    )

    print_result_info(result, "Initial Result")

    if result.finish_reason == "length":
        print("\n🔄 Continuing generation (max_continues=3)...")
        # Use max_continues=3 to allow multiple continuation attempts
        full_result = ChatContinue.continue_request(
            chat,
            result,
            history=history,
            max_continues=3,
        )

        print_result_info(full_result, "After Multiple Continues")
        print(f"Full text length: {len(full_result.text)} characters")

        if full_result.finish_reason == "length":
            print(
                "⚠️  Still truncated after 3 continues - may need more continues or larger max_tokens"
            )
        else:
            print("✅ Successfully completed after multiple continues")


def scenario_3_streaming_continue(chat: Chat):
    """Scenario 3: Streaming continue."""
    print_section("Scenario 3: Streaming Continue")

    history = ChatHistory()

    print("Requesting a long response with streaming and small max_tokens...")
    # First, get initial result with streaming
    print("\n📡 Initial streaming response:")
    initial_iterator = chat.stream(
        "Write a detailed explanation of machine learning algorithms. Cover at least: supervised learning, unsupervised learning, and reinforcement learning. Make it comprehensive.",
        history=history,
        max_tokens=40,  # Small to force truncation
    )

    initial_text = ""
    for chunk in initial_iterator:
        print(chunk.delta, end="", flush=True)
        initial_text += chunk.delta

    print("\n")
    initial_result = initial_iterator.result.to_chat_result()
    print_result_info(initial_result, "Initial Streaming Result")

    if initial_result.finish_reason == "length":
        print("\n🔄 Continuing with streaming...")
        continue_iterator = ChatContinue.continue_request_stream(
            chat,
            initial_result,
            history=history,
            max_continues=2,
        )

        print("📡 Continue streaming response:")
        for chunk in continue_iterator:
            print(chunk.delta, end="", flush=True)

        print("\n")
        final_result = continue_iterator.result.to_chat_result()
        print_result_info(final_result, "Final Streaming Result")

        if final_result.finish_reason == "length":
            print("⚠️  Still truncated after streaming continue")
        else:
            print("✅ Successfully completed with streaming continue")


def scenario_4_manual_continue(chat: Chat):
    """Scenario 4: Manual continue without auto_merge (get all intermediate results)."""
    print_section("Scenario 4: Manual Continue (Without Auto-Merge)")

    history = ChatHistory()

    print("Requesting a long response and manually handling each continue result...")
    result = chat(
        "Write a detailed article about renewable energy sources. Cover solar, wind, hydroelectric, and geothermal energy. Make it at least 800 words.",
        history=history,
        max_tokens=35,  # Small to force truncation
    )

    print_result_info(result, "Initial Result")

    if result.finish_reason == "length":
        print("\n🔄 Continuing with auto_merge=False to get all intermediate results...")
        # Get all results separately
        all_results = ChatContinue.continue_request(
            chat,
            result,
            history=history,
            max_continues=2,
            auto_merge=False,
        )

        print(f"\n📊 Received {len(all_results)} results:")
        total_length = 0
        for i, res in enumerate(all_results):
            print_result_info(res, f"Result {i + 1}")
            total_length += len(res.text)

        print(f"Total text length across all results: {total_length} characters")

        # Manually merge if needed
        if len(all_results) > 1:
            print("\n🔗 Manually merging results...")
            merged = ChatContinue.merge_results(*all_results)
            print_result_info(merged, "Manually Merged Result")


def scenario_5_custom_continue_prompt(chat: Chat):
    """Scenario 5: Continue with custom prompts."""
    print_section("Scenario 5: Continue with Custom Prompts")

    history = ChatHistory()

    print("Requesting a long response and continuing with custom prompt...")
    result = chat(
        "Write a comprehensive tutorial on web development. Cover HTML, CSS, JavaScript, and modern frameworks. Make it detailed and at least 1000 words.",
        history=history,
        max_tokens=40,  # Small to force truncation
    )

    print_result_info(result, "Initial Result")

    if result.finish_reason == "length":
        print("\n🔄 Continuing with custom prompt: 'Please continue from where you left off.'...")
        # Use a custom continue prompt
        full_result = ChatContinue.continue_request(
            chat,
            result,
            history=history,
            max_continues=2,
            continue_prompt="Please continue from where you left off.",
        )

        print_result_info(full_result, "After Custom Continue")

        # Test without adding continue prompt (direct continuation)
        if full_result.finish_reason == "length":
            print("\n🔄 Trying continue without adding prompt (add_continue_prompt=False)...")
            result2 = ChatContinue.continue_request(
                chat,
                full_result,
                history=history,
                max_continues=1,
                add_continue_prompt=False,
            )
            print_result_info(result2, "Continue Without Prompt")


def scenario_6_large_input_small_output(chat: Chat):
    """
    Scenario 6: Large input + small output limit.

    This simulates production scenarios where:
    - Models have large input context windows
    - But small output token limits
    - Need to continue multiple times to get full response
    """
    print_section("Scenario 6: Large Input + Small Output Limit (Production Scenario)")

    history = ChatHistory()

    # Create a large input (simulating large context)
    large_input = "Here is a detailed document about artificial intelligence:\n\n"
    large_input += (
        "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. "
        * 20
    )
    large_input += "\n\nBased on the above document, please write a comprehensive summary covering all key points. Make it detailed and thorough."

    print(f"Input size: {len(large_input)} characters")
    print("Requesting response with very small max_tokens (simulating small output limit)...")

    result = chat(
        large_input,
        history=history,
        max_tokens=25,  # Very small output limit
    )

    print_result_info(result, "Initial Result")
    print(f"Input tokens: {result.usage.input_tokens}, Output tokens: {result.usage.output_tokens}")

    if result.finish_reason == "length":
        print("\n🔄 Continuing multiple times to get full response...")
        # Continue multiple times
        full_result = ChatContinue.continue_request(
            chat,
            result,
            history=history,
            max_continues=5,  # Allow many continues
        )

        print_result_info(full_result, "After Multiple Continues")
        print(f"Total output tokens: {full_result.usage.output_tokens}")

        if full_result.finish_reason == "length":
            print("⚠️  Still truncated - output limit is very restrictive")
        else:
            print("✅ Successfully got complete response despite small output limit")


def scenario_7_concatenate_and_continue(chat: Chat):
    """
    Scenario 7: Concatenate output back to input and continue.

    This tests a production pattern where:
    - After interruption, we concatenate the output to the input (as additional context)
    - Then make a new chat request to get more output
    - This is a manual pattern that some users might use when working with models
      that have large input windows but small output limits
    - Compare this with the library's automatic continue mechanism
    """
    print_section("Scenario 7: Concatenate Output to Input and Continue (Production Pattern)")

    # Test 1: Manual concatenation pattern
    print("Test 1: Manual concatenation pattern (output -> input context)")
    history1 = ChatHistory()

    original_prompt = "Write a detailed guide on database optimization techniques. Cover indexing, query optimization, and caching strategies. Make it comprehensive and at least 1200 words."

    print("Requesting initial response with small max_tokens...")
    result1 = chat(
        original_prompt,
        history=history1,
        max_tokens=30,  # Small to force truncation
    )

    print_result_info(result1, "Initial Result")

    if result1.finish_reason == "length":
        print("\n🔄 Manual Pattern: Concatenating output to input and making new request...")

        # Manual pattern: concatenate output to input
        partial_output = result1.text
        print(f"Partial output length: {len(partial_output)} characters")

        # Create new prompt with concatenated output as context
        new_prompt = f"{original_prompt}\n\nHere is what has been written so far:\n{partial_output}\n\nPlease continue writing from where it left off."

        # Make a new chat request (not using continue mechanism)
        history2 = ChatHistory()  # New history for this pattern
        continue_result = chat(
            new_prompt,
            history=history2,
            max_tokens=30,  # Still small
        )

        print_result_info(continue_result, "Continue Result (Manual Pattern)")

        # Manually merge
        merged_text = partial_output + continue_result.text
        print(f"\n🔗 Manually merged text length: {len(merged_text)} characters")

        if continue_result.finish_reason == "length":
            print("⚠️  Continue result also truncated - pattern works but needs iteration")
        else:
            print("✅ Successfully continued using manual concatenation pattern")

    # Test 2: Compare with library's automatic continue
    print("\n" + "-" * 80)
    print("Test 2: Compare with library's automatic continue mechanism")
    print("-" * 80)

    history3 = ChatHistory()
    result3 = chat(
        original_prompt,
        history=history3,
        max_tokens=30,  # Same small limit
    )

    if result3.finish_reason == "length":
        print("\n🔄 Using library's automatic continue mechanism...")
        auto_result = ChatContinue.continue_request(
            chat,
            result3,
            history=history3,
            max_continues=2,
        )

        print_result_info(auto_result, "Auto Continue Result")
        print(f"\n📊 Comparison:")
        print(
            f"  Manual pattern final length: {len(merged_text) if result1.finish_reason == 'length' else 'N/A'} chars"
        )
        print(f"  Auto continue final length: {len(auto_result.text)} chars")
        print(f"\n💡 Recommendation: Use library's continue mechanism for better integration")


def scenario_8_error_handling(chat: Chat):
    """Scenario 8: Error handling and edge cases."""
    print_section("Scenario 8: Error Handling and Edge Cases")

    history = ChatHistory()

    # Test 1: Try to continue a result that's not truncated
    print("Test 1: Attempting to continue a non-truncated result...")
    result = chat(
        "Say hello in one sentence.",
        history=history,
        max_tokens=100,  # Large enough to not truncate
    )

    print_result_info(result, "Non-truncated Result")

    if result.finish_reason != "length":
        print("Attempting continue_request on non-truncated result (should raise ValueError)...")
        try:
            ChatContinue.continue_request(chat, result, history=history)
            print("❌ ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"✅ Correctly raised ValueError: {e}")

    # Test 2: Continue without history
    print("\nTest 2: Attempting continue without history...")
    result2 = chat(
        "Write a short story.",
        max_tokens=20,  # Small to force truncation
    )

    if result2.finish_reason == "length":
        print("Attempting continue_request without history (should raise ValueError)...")
        try:
            ChatContinue.continue_request(chat, result2, history=None)
            print("❌ ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"✅ Correctly raised ValueError: {e}")

    # Test 3: Continue with empty result
    print("\nTest 3: Edge case - result with empty text but length finish_reason...")
    # This is unlikely but we should handle it
    print("(This scenario is rare but demonstrates robustness)")


def scenario_9_complete_method(chat: Chat):
    """Scenario 9: Using Chat.complete() method (recommended approach)."""
    print_section("Scenario 9: Using Chat.complete() (Recommended Approach)")

    history = ChatHistory()

    print("Using Chat.complete() to ensure complete response...")
    print("This is the recommended method for production use.")

    try:
        result = chat.complete(
            "Write a comprehensive article about cloud computing. Cover IaaS, PaaS, SaaS, and modern cloud architectures. Make it detailed and at least 1500 words.",
            history=history,
            max_tokens=35,  # Small to force truncation
            max_continues=3,
            ensure_complete=True,  # Will raise error if still truncated
        )

        print_result_info(result, "Complete Result")
        print("✅ Successfully got complete response using chat.complete()")

    except Exception as e:
        print(
            f"⚠️  Exception (expected if still truncated after max_continues): {type(e).__name__}: {e}"
        )
        print("This is expected behavior when ensure_complete=True and response is still truncated")

    # Test with ensure_complete=False
    print("\nTesting with ensure_complete=False (allows partial result)...")
    history2 = ChatHistory()
    result2 = chat.complete(
        "Write a very long article about quantum computing.",
        history=history2,
        max_tokens=25,  # Very small
        max_continues=2,  # Limited continues
        ensure_complete=False,  # Allow partial result
    )

    print_result_info(result2, "Complete Result (ensure_complete=False)")
    if result2.finish_reason == "length":
        print("⚠️  Result is still truncated (allowed because ensure_complete=False)")


def scenario_10_streaming_complete(chat: Chat):
    """Scenario 10: Using Chat.complete_stream() method."""
    print_section("Scenario 10: Using Chat.complete_stream() (Streaming + Complete)")

    history = ChatHistory()

    print("Using Chat.complete_stream() for streaming with automatic continuation...")

    try:
        iterator = chat.complete_stream(
            "Write a detailed tutorial on REST API design. Cover endpoints, HTTP methods, status codes, authentication, and best practices. Make it comprehensive and at least 2000 words.",
            history=history,
            max_tokens=40,  # Small to force truncation
            max_continues=2,
            ensure_complete=True,
        )

        print("\n📡 Streaming complete response:")
        for chunk in iterator:
            print(chunk.delta, end="", flush=True)

        print("\n")
        result = iterator.result.to_chat_result()
        print_result_info(result, "Complete Streaming Result")
        print("✅ Successfully got complete response using chat.complete_stream()")

    except Exception as e:
        print(f"\n⚠️  Exception (expected if still truncated): {type(e).__name__}: {e}")


def main():
    """Run all continue chat scenarios."""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading configuration: {e}")
        print("\nUsing default placeholder values. To use real API:")
        print("  1. Create tests/test_endpoints.json with your API credentials")
        print(
            "  2. Or specify a config file: python examples/continue_chat_demo.py --config /path/to/config.json"
        )
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    # Initialize chat client
    chat = Chat(**config)

    print("\n" + "=" * 80)
    print("  COMPREHENSIVE CONTINUE CHAT DEMO")
    print("  Testing various continue chat scenarios with small max_tokens")
    print("=" * 80)

    # Run all scenarios
    scenarios = [
        scenario_1_basic_continue,
        scenario_2_multiple_continues,
        scenario_3_streaming_continue,
        scenario_4_manual_continue,
        scenario_5_custom_continue_prompt,
        scenario_6_large_input_small_output,
        scenario_7_concatenate_and_continue,
        scenario_8_error_handling,
        scenario_9_complete_method,
        scenario_10_streaming_complete,
    ]

    for i, scenario in enumerate(scenarios, 1):
        try:
            scenario(chat)
        except Exception as e:
            print(f"\n❌ Error in scenario {i}: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("  DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
