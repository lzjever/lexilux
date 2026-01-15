#!/usr/bin/env python
"""
Error Handling Demo (Updated for Lexilux v2.1+)

Demonstrates how to handle errors using the new Lexilux exception hierarchy.
Shows how to use custom exceptions with error codes and retryable flags.
"""

from lexilux import (
    AuthenticationError,
    Chat,
    LexiluxError,
    RateLimitError,
    TimeoutError,
)
from lexilux import (
    ConnectionError as LexiluxConnectionError,
)


def demo_non_streaming_error_handling():
    """Demonstrate error handling for non-streaming requests with new exceptions"""
    print("=" * 70)
    print("Demo: Non-Streaming Error Handling (New Exceptions)")
    print("=" * 70)

    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-api-key",
        model="gpt-4",
    )

    try:
        result = chat("Hello, world!")
        # Success: finish_reason indicates why generation stopped
        print(f"✓ Success: finish_reason = {result.finish_reason}")
        print(f"  Text: {result.text[:50]}...")
        print(f"  Tokens: {result.usage.total_tokens}")

        # Check finish_reason to understand why it stopped
        if result.finish_reason == "stop":
            print("  → Normal completion (stopped naturally)")
        elif result.finish_reason == "length":
            print("  → Hit max_tokens limit")
        elif result.finish_reason == "content_filter":
            print("  → Content was filtered")
        elif result.finish_reason is None:
            print("  → Unknown reason (API didn't provide finish_reason)")

    except LexiluxConnectionError as e:
        # Connection failed
        print(f"✗ Connection Error: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        print("  → Network problem: Could not connect to server")
        if e.retryable:
            print("  → Suggestion: Retry the request")

    except TimeoutError as e:
        # Request timeout
        print(f"✗ Timeout Error: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        print("  → Request took too long to complete")
        print("  → Suggestion: Increase timeout_s or retry")

    except AuthenticationError as e:
        # Authentication failed (401)
        print(f"✗ Authentication Error: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        print("  → API key is invalid or expired")
        print("  → Suggestion: Check your API key")

    except RateLimitError as e:
        # Rate limit exceeded (429)
        print(f"✗ Rate Limit Error: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        print("  → Too many requests")
        print("  → Suggestion: Wait and retry, or upgrade your plan")

    except LexiluxError as e:
        # Catch-all for any other Lexilux errors
        print(f"✗ API Error: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        print("  → An API error occurred")


def demo_streaming_error_handling():
    """Demonstrate error handling for streaming requests with new exceptions"""
    print("\n" + "=" * 70)
    print("Demo: Streaming Error Handling (New Exceptions)")
    print("=" * 70)

    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-api-key",
        model="gpt-4",
    )

    chunks = []
    completed = False
    finish_reason = None

    try:
        print("Streaming response:")
        print("-" * 50)
        for chunk in chat.stream("Write a long story about programming"):
            print(chunk.delta, end="", flush=True)
            chunks.append(chunk)

            # Track if we received a completion
            if chunk.done:
                completed = True
                # Find chunk with finish_reason (may be before [DONE])
                done_chunks = [c for c in chunks if c.done]
                final_chunk = next(
                    (c for c in done_chunks if c.finish_reason is not None),
                    done_chunks[-1] if done_chunks else None,
                )
                if final_chunk:
                    finish_reason = final_chunk.finish_reason

        print("\n" + "-" * 50)

        if completed:
            print(f"✓ Stream completed: finish_reason = {finish_reason}")
            if finish_reason == "stop":
                print("  → Normal completion (stopped naturally)")
            elif finish_reason == "length":
                print("  → Hit max_tokens limit")
            elif finish_reason == "content_filter":
                print("  → Content was filtered")
            elif finish_reason is None:
                print("  → Unknown reason (API sent [DONE] without finish_reason)")
        else:
            print("⚠ Stream ended without completion signal")
            print("  → This shouldn't happen in normal operation")

    except LexiluxConnectionError as e:
        print(f"\n✗ Connection Error during streaming: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        if completed:
            print(f"  → Completion occurred before error: finish_reason = {finish_reason}")
        else:
            print("  → No completion received - stream was interrupted")

    except TimeoutError as e:
        print(f"\n✗ Timeout Error during streaming: {e.message}")
        print(f"  Error code: {e.code}")
        print(f"  Retryable: {e.retryable}")
        if completed:
            print(f"  → Completion occurred before timeout: finish_reason = {finish_reason}")
        else:
            print("  → No completion received - stream timed out")


def demo_error_code_inspection():
    """Demonstrate error code and retryable flag inspection"""
    print("\n" + "=" * 70)
    print("Demo: Error Code and Retryable Flag Inspection")
    print("=" * 70)

    # Example client configuration (not used in this demo)
    _ = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-api-key",
        model="gpt-4",
    )

    # Example: Create chat with retry enabled
    _ = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-api-key",
        max_retries=3,  # Automatically retry retryable errors
    )

    print("\nExample Error Codes and Meanings:")
    print("-" * 50)
    error_examples = [
        (AuthenticationError, "authentication_failed", False, "Invalid API key"),
        (RateLimitError, "rate_limit_exceeded", True, "Too many requests"),
        (TimeoutError, "timeout", True, "Request timeout"),
        (LexiluxConnectionError, "connection_failed", True, "Network failure"),
    ]

    for exc_class, code, retryable, description in error_examples:
        exc = exc_class(description)
        print(f"  {exc.__class__.__name__}:")
        print(f"    Code: {exc.code}")
        print(f"    Retryable: {exc.retryable}")
        print(f"    Description: {description}")
        print()


def demo_retry_logic_with_retryable_flag():
    """Demonstrate using retryable flag for custom retry logic"""
    print("\n" + "=" * 70)
    print("Demo: Custom Retry Logic Using Retryable Flag")
    print("=" * 70)

    import time

    chat = Chat(
        base_url="https://api.example.com/v1",
        api_key="your-api-key",
        model="gpt-4",
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            _ = chat("Hello, world!")
            print(f"✓ Success on attempt {attempt + 1}")
            break
        except LexiluxError as e:
            print(f"  Attempt {attempt + 1} failed: {e.code} - {e.message}")

            if e.retryable and attempt < max_retries - 1:
                # Retry retryable errors
                wait_time = 2**attempt  # Exponential backoff
                print(f"  → Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Don't retry non-retryable errors or on last attempt
                print("  → Not retryable or max retries reached")
                raise


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("Error Handling Demo (Lexilux v2.1+)")
    print("=" * 70)
    print("\nThis demo shows how to use the new Lexilux exception hierarchy:")
    print("  1. Specific exception types for different errors")
    print("  2. Error codes for programmatic handling")
    print("  3. Retryable flag for automatic retry logic")
    print("\n" + "=" * 70)

    demo_non_streaming_error_handling()
    demo_streaming_error_handling()
    demo_error_code_inspection()
    demo_retry_logic_with_retryable_flag()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Use specific exceptions (AuthenticationError, RateLimitError, etc.)")
    print("2. Check error.code for programmatic error handling")
    print("3. Check error.retryable before implementing retry logic")
    print("4. Enable max_retries for automatic retry on retryable errors")
    print("5. LexiluxError is the base exception for all Lexilux errors")
    print("=" * 70)


if __name__ == "__main__":
    main()
