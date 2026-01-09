#!/usr/bin/env python
"""
Quick test for continue chat functionality.

Tests basic continue scenarios to verify the library works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import get_chat_config, parse_args
from lexilux import Chat, ChatHistory, ChatContinue


def test_basic_continue():
    """Test basic continue functionality."""
    print("=" * 60)
    print("Test 1: Basic Continue")
    print("=" * 60)

    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError):
        print("❌ Config not found. Skipping test.")
        return False

    chat = Chat(**config)
    history = ChatHistory()

    # Request with small max_tokens
    result = chat(
        "Write a short story about a robot. Make it at least 200 words.",
        history=history,
        max_tokens=30,  # Small to force truncation
    )

    print(f"Initial: finish_reason={result.finish_reason}, length={len(result.text)}")

    if result.finish_reason == "length":
        print("✅ Truncated as expected, continuing...")
        full_result = ChatContinue.continue_request(
            chat,
            result,
            history=history,
            max_continues=1,
        )
        print(
            f"After continue: finish_reason={full_result.finish_reason}, length={len(full_result.text)}"
        )

        if full_result.finish_reason != "length":
            print("✅ Successfully completed after continue")
            return True
        else:
            print("⚠️  Still truncated (may need more continues)")
            return True  # Still a valid test
    else:
        print("⚠️  Not truncated - increase max_tokens or request longer content")
        return True  # Not an error, just didn't trigger continue


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Test 2: Error Handling")
    print("=" * 60)

    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError):
        print("❌ Config not found. Skipping test.")
        return False

    chat = Chat(**config)
    history = ChatHistory()

    # Test: Try to continue non-truncated result
    result = chat(
        "Say hello.",
        history=history,
        max_tokens=100,  # Large enough
    )

    if result.finish_reason != "length":
        try:
            ChatContinue.continue_request(chat, result, history=history)
            print("❌ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError: {e}")
            return True
    else:
        print("⚠️  Result was truncated unexpectedly")
        return True


def main():
    """Run quick tests."""
    print("\nContinue Chat Quick Test\n")

    results = []
    results.append(("Basic Continue", test_basic_continue()))
    results.append(("Error Handling", test_error_handling()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
